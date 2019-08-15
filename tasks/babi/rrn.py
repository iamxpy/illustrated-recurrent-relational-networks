import multiprocessing as mp
import os
import queue
import random
import threading
import time

import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import LSTMCell

import util
from message_passing import message_passing
from model import Model
from tasks.babi.data import bAbI


class BaBiRecurrentRelationalNet(Model):
    devices = util.get_devices()
    revision = os.environ.get('REVISION')
    message = os.environ.get('MESSAGE')
    name = "%s %s" % (revision, message)
    emb_size = 32
    batch_size = 160
    num_facts = 20  # only use the last 20 facts
    qsize = len(devices) * 100
    n_steps = 5
    edge_keep_prob = 1.0
    n_hidden = 128

    def __init__(self, is_testing):
        super().__init__()
        self.is_testing = is_testing

        print("Preparing data...")
        # Load and encode data (Disk -> Memory), see more details in encode_data()
        # Also see data_loader(), the next processing stage.
        self.train, self.valid, self.test, self.vocab = self.encode_data(bAbI('en-valid-10k'))

        print("Creating graph...")
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            regularizer = layers.l2_regularizer(1e-4)  # regularizer applied to fully-connected network

            # allow_soft_placement=True: if cannot find specific device, allow tf to choose the device
            self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)

            self.facts_ph = tf.placeholder(tf.int32, shape=(None, None))  # (bs*#facts, seq)
            self.facts_pos_ph = tf.placeholder(tf.int32, shape=(None,))  # (bs*#facts, )
            self.question_ph = tf.placeholder(tf.int32, shape=(None, None))  # (bs, seq)
            self.answers_ph = tf.placeholder(tf.int32, shape=(None,))  # (bs, )
            self.edge_indices_ph = tf.placeholder(tf.int32, shape=(None, 2))
            self.fact_segments_ph = tf.placeholder(tf.int32, shape=(None,))
            self.edge_segments_ph = tf.placeholder(tf.int32, shape=(None,))
            self.q_seq_length_ph = tf.placeholder(tf.int32, shape=(None,))
            self.f_seq_length_ph = tf.placeholder(tf.int32, shape=(None,))
            self.task_indices_ph = tf.placeholder(tf.int32, shape=(None,))
            self.edge_keep_prob_ph = tf.placeholder(tf.float32, shape=())
            self.is_training_ph = tf.placeholder(tf.bool)

            # device: CPU:0
            placeholders = [self.facts_ph, self.facts_pos_ph, self.question_ph, self.answers_ph, self.edge_indices_ph,
                            self.fact_segments_ph, self.edge_segments_ph, self.q_seq_length_ph, self.f_seq_length_ph,
                            self.task_indices_ph, self.edge_keep_prob_ph]

            # each element of train_queue is a training batch
            self.train_queue = tf.FIFOQueue(self.qsize, [ph.dtype for ph in placeholders], name='train-queue')
            # each element of train_queue is a validation batch
            self.val_queue = tf.FIFOQueue(self.qsize, [ph.dtype for ph in placeholders], name='val-queue')

            self.train_enqueue_op = self.train_queue.enqueue(placeholders)
            self.train_qsize_op = self.train_queue.size()
            # record the size of the train_queue every batch
            tf.summary.scalar('queues/train', self.train_qsize_op)

            self.val_enqueue_op = self.val_queue.enqueue(placeholders)
            self.val_qsize_op = self.val_queue.size()
            # record the size of the val_queue every batch
            tf.summary.scalar('queues/val', self.val_qsize_op)

            def avg_n(x):
                return tf.reduce_mean(tf.stack(x, axis=0), axis=0)

            towers = []
            with tf.variable_scope(tf.get_variable_scope()):
                for device_nr, device in enumerate(self.devices):
                    with tf.device('/cpu:0'):
                        if self.is_testing:
                            facts_ph, facts_pos_ph, question_ph, answers_ph, edge_indices_ph, fact_segments_ph, edge_segments_ph, q_seq_length_ph, f_seq_length_ph, task_indices_ph, edge_keep_prob = placeholders
                        else:
                            facts_ph, facts_pos_ph, question_ph, answers_ph, edge_indices_ph, fact_segments_ph, edge_segments_ph, q_seq_length_ph, f_seq_length_ph, task_indices_ph, edge_keep_prob = tf.cond(
                                self.is_training_ph,
                                true_fn=lambda: self.train_queue.dequeue(),
                                false_fn=lambda: self.val_queue.dequeue(),
                            )
                            # device: CPU:0, CPU:0, CPU:0 (In a 3 GPU machine, these placeholders are in triplicate.)
                            vars = (facts_ph, facts_pos_ph, question_ph, answers_ph, edge_indices_ph, fact_segments_ph,
                                    edge_segments_ph, q_seq_length_ph, f_seq_length_ph, task_indices_ph, edge_keep_prob)

                            for v, ph in zip(vars, placeholders):
                                v.set_shape(ph.get_shape())
                        # device: CPU:0, CPU:0, CPU:0
                        facts_emb = layers.embed_sequence(facts_ph, self.vocab.size(), self.emb_size,
                                                          scope='word-embeddings')
                        # device: CPU:0, CPU:0, CPU:0
                        questions_emb = layers.embed_sequence(question_ph, self.vocab.size(), self.emb_size,
                                                              scope='word-embeddings', reuse=True)

                    with tf.device(device), tf.name_scope("device-%s" % device_nr):
                        # 4 layers FC
                        def mlp(x, scope, n_hidden):
                            with tf.variable_scope(scope):
                                for i in range(3):
                                    x = layers.fully_connected(x, n_hidden, weights_regularizer=regularizer)
                                return layers.fully_connected(x, n_hidden, weights_regularizer=regularizer,
                                                              activation_fn=None)

                        # get the final hidden state for the sentences(facts), f_encoding shape: (bs*#facts, state_size)
                        _, (_, f_encoding) = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(32), facts_emb, dtype=tf.float32,
                                                               sequence_length=f_seq_length_ph, scope='fact-encoder')

                        # shape:(bs, ) (the same as answers_ph), elements inside the vector range from 0 to 20 randomly
                        # and subjects to the normal distribution
                        random_pos_offsets = tf.random_uniform(tf.shape(answers_ph), minval=0, maxval=self.num_facts,
                                                               dtype=tf.int32)
                        # Generate random offset. Note that for a specific task, the offset is the same.
                        fact_pos = facts_pos_ph + tf.gather(random_pos_offsets, fact_segments_ph)
                        # Considering the offset, the depth for the positional one-hot encoding should be 2*num_facts
                        facts_pos_encoding = tf.one_hot(fact_pos, 2 * self.num_facts)

                        # concatenate the encoding of content and position; device: GPU:0, GPU:1, GPU:2
                        f_encoding = tf.concat([f_encoding, facts_pos_encoding], axis=1)

                        # Need not to encode position for questions, just get the features of their content
                        # q_encoding shape: (bs, state_size); device: GPU:0, GPU:1, GPU: 2
                        _, (_, q_encoding) = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(32), questions_emb,
                                                               dtype=tf.float32, sequence_length=q_seq_length_ph,
                                                               scope='question-encoder')

                        # MLP of 3 layers FC, used to process the output of a graph
                        # num output of last layer is vocab.size(), so as to get the logits
                        def graph_fn(x):
                            with tf.variable_scope('graph-fn'):
                                x = layers.fully_connected(x, self.n_hidden, weights_regularizer=regularizer)
                                x = layers.fully_connected(x, self.n_hidden, weights_regularizer=regularizer)
                                return layers.fully_connected(x, self.vocab.size(), activation_fn=None,
                                                              weights_regularizer=regularizer)

                        # concatenate the fact_encoding and the question_encoding
                        x = tf.concat([f_encoding, tf.gather(q_encoding, fact_segments_ph)], 1)

                        # x0 represents "fact embedding given the question"
                        # (by concatenate the question embedding with them)
                        # device: GPU:0, GPU:1, GPU:2
                        x0 = mlp(x, 'pre', self.n_hidden)

                        # generate the question encoding for every edge
                        # edge_features shape: (bs*(#facts**2), LSTM state_size)
                        edge_features = tf.gather(q_encoding, edge_segments_ph)

                        x = x0
                        outputs = []
                        log_losses = []
                        with tf.variable_scope('steps'):
                            lstm_cell = LSTMCell(self.n_hidden)
                            state = lstm_cell.zero_state(tf.shape(x)[0], tf.float32)

                            for step in range(self.n_steps):
                                x = message_passing(x, edge_indices_ph, edge_features,
                                                    lambda x: mlp(x, 'message-fn', self.n_hidden), edge_keep_prob)

                                x = mlp(tf.concat([x, x0], axis=1), 'post-fn', self.n_hidden)
                                # x=hidden state, state=<cell state, hidden state>
                                # device: (GPU:0)*5, (GPU:1)*5, (GPU:2)*5 (5 is the time step)
                                x, state = lstm_cell(x, state)
                                with tf.variable_scope('graph-sum'):
                                    # In every step, get the sum of output vectors of Nodes for every task(Graph)
                                    # i.e. graph_sum shape: (bs, n_hidden)
                                    graph_sum = tf.segment_sum(x, fact_segments_ph)
                                    out = graph_fn(graph_sum)  # shape: (bs, vocab_size)
                                    outputs.append(out)
                                    # softmax loss, scalar Tensor
                                    log_loss=tf.reduce_mean(
                                        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers_ph, logits=out))
                                    # log_losses is a list of scalar Tensor, each one means the loss in a time step
                                    log_losses.append(log_loss)

                                # reuse the Variables in LSTM across different time step
                                tf.get_variable_scope().reuse_variables()
                        # scalr Tensor, the sum of all regularization term loss
                        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        # avg_n(log_losses) gets the mean loss for every step, i.e. "loss" is a scalar Tensor
                        # device: GPU:0, GPU:1, GPU:2
                        loss = avg_n(log_losses) + reg_loss

                        # device: GPU:0, GPU:1, GPU:2
                        stat={
                            'loss': loss,  # scalar Tensor
                            'grads': self.optimizer.compute_gradients(loss),
                            'log_losses': tf.stack(log_losses),  # (n_steps, )
                            'answers': answers_ph,  # (batch_size, )
                            'outputs': tf.stack(outputs),  # (n_steps, batch_size, vocab_size)
                            'task_indices': task_indices_ph  # (batch_size, )
                        }
                        towers.append(stat)
                        print('line 159: ')
                        print('"' + tf.get_variable_scope().name + '"')
                        # reuse the Variables in embedding, encoder, and some MLPs across different device
                        tf.get_variable_scope().reuse_variables()

            # device of the following 4 vars is CPU:0
            self.loss = avg_n([t['loss'] for t in towers])
            self.out = tf.concat([t['outputs'] for t in towers], axis=1)
            self.answers = tf.concat([t['answers'] for t in towers], axis=0)
            self.task_indices = tf.concat([t['task_indices'] for t in towers], axis=0)

            tf.summary.scalar('losses/total', self.loss)
            tf.summary.scalar('losses/reg', reg_loss)
            log_losses = avg_n([t['log_losses'] for t in towers])
            for i in range(self.n_steps):
                tf.summary.scalar('steps/%d/losses/log' % i, log_losses[i])

            avg_gradients = util.average_gradients([t['grads'] for t in towers])

            # global_step increases by 1 after the gradient is updated
            self.train_step = self.optimizer.apply_gradients(avg_gradients, global_step=self.global_step)

            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            util.print_vars(tf.trainable_variables())

            self.train_writer = tf.summary.FileWriter('/tmp/tensorboard/bAbI/%s/train/%s' % (self.revision, self.name),
                                                      self.session.graph)
            self.test_writer = tf.summary.FileWriter('/tmp/tensorboard/bAbI/%s/test/%s' % (self.revision, self.name),
                                                     self.session.graph)

            self.summaries = tf.summary.merge_all()

        print("Starting data loaders...")
        train_mp_queue = mp.Manager().Queue(maxsize=self.qsize)
        val_mp_queue = mp.Manager().Queue(maxsize=self.qsize)

        # After loaded data from disk(done in the code `self.encode_data(bAbI('en-valid-10k'))`),
        # use 4+1=5 Processes to construct batches and encode them, then enqueue them onto corresponding queue.
        # see more details in random_batch() and encode_batch()
        data_loader_processes = [mp.Process(target=self.data_loader, args=(train_mp_queue, True)) for i in range(4)]
        val_data_loader_processes = [mp.Process(target=self.data_loader, args=(val_mp_queue, False)) for i in range(1)]

        # start the processes
        for p in data_loader_processes + val_data_loader_processes:
            p.daemon = True
            p.start()

        # Use 2 threads to transfer data from train_mp_queue(val_mp_queue) to train_queue(val_queue).
        # Note that batch in train_mp_queue is ndarray of numpy,
        # and these two thread change every batch into Tensors and enqueue it onto train_queue.
        # see the placeholders defined before for the format of each batch.
        queue_putter_threads = [
            threading.Thread(target=self.queue_putter, args=(train_mp_queue, self.train_enqueue_op, 'train', 1000)),
            threading.Thread(target=self.queue_putter, args=(val_mp_queue, self.val_enqueue_op, 'val', 1)),
        ]
        # start data transferring
        for t in queue_putter_threads:
            t.daemon = True
            t.start()

        train_qsize, val_qsize = 0, 0
        print("Waiting for queue to fill...")
        while train_qsize < self.qsize or val_qsize < self.qsize:
            # update the size of the queues of training and validation
            train_qsize = self.session.run(self.train_qsize_op)
            val_qsize = self.session.run(self.val_qsize_op)
            print('train_qsize: %d, val_qsize: %d' % (train_qsize, val_qsize), flush=True)
            time.sleep(1)

    def data_loader(self, queue: mp.Queue, is_training):
        while True:
            # TODO 1. For every iteration, get batch from get_batch(), pass it to get_feed_dict()
            # TODO 2. Feed the resulting dict as param directly to session.run() in train_batch()
            queue.put(self.get_batch(is_training))

    def queue_putter(self, q: mp.Queue, enqueue_op, name, print_interval):
        total = 0
        starved = 0
        start = time.time()

        while True:
            try:
                batch = q.get_nowait()
                self.session.run(enqueue_op, self.get_feed_dict(batch))
            except queue.Empty:
                starved += 1
                time.sleep(0.1)

            total += 1
            if total % print_interval == 0:
                took = time.time() - start
                print("%s %f batches/s, %d starved, %d total, qsize %d" % (name, total / took, starved, total, q.qsize()))


    # P.S. #facts here is not a fixed number, it means the number of the facts of every task.
    # e.g. If a batch has 2 task, the first one has 3 facts, the second one has 2 facts, #facts for this task is 2+3=5
    def get_feed_dict(self, batch):
        facts, fact_positions, f_seq_length, questions, q_seq_length, answers, fact_segments, edge_indices, edge_segments, task_indices, edge_keep_prob = batch
        feed_dict = {
            self.facts_ph: facts,  # (bs*#facts, seq)
            # position of fact is used to construct part of the embedding of fact
            self.facts_pos_ph: fact_positions,  # (bs*#facts, )
            # record length of every fact, because we will pad them later
            self.f_seq_length_ph: f_seq_length,  # (bs*#facts, )

            self.question_ph: questions,  # (bs, seq)
            self.q_seq_length_ph: q_seq_length,  # (bs, )

            self.answers_ph: answers,  # (bs, )
            self.fact_segments_ph: fact_segments,  # (bs*#facts, )

            self.edge_indices_ph: edge_indices,  # (bs*(#facts**2), 2)
            self.edge_segments_ph: edge_segments, # (bs*(#facts**2), )

            self.task_indices_ph: task_indices, # (bs, )
            self.edge_keep_prob_ph: edge_keep_prob # 1
        }
        return feed_dict

    # vocab acts like a map (from word to index), see more details in class Vocab
    def get_vocab(self, tasks):
        words = set()
        # TODO this for loop is too slow, make it less so
        for task in tasks:  # extract unique words from training corpus
            for question in task:
                words.add(question['a'])

                for qw in question['q'].split(' '):
                    words.add(qw)

                for fact in question['facts']:
                    for fw in fact.split(' '):
                        words.add(fw)

        sorted_words = sorted(list(words))  # sort the words
        vocab = {w: i for i, w in enumerate(sorted_words)}  # <word, id>
        return Vocab(vocab, "UNK")

    '''
    For a specific learning stage (training/validation/test)：
    BABI corpus contains 20 types of tasks, each of which corresponds to one file(e.g. qa1_train.txt)
    "tasks" is a list with length 20, as mentioned before, each element is a type of task and contains content from a file
    The content of a file can be split into many specific task by the questions, e.g. 3th and 6th line in qa1_train.txt
    
    Each specific task will be transformed into following content:
    1. facts: 2-D matrix of vocabulary indices of facts, e.g. [[94, 87, 123, 117, 45],[...]] represents [[Mary moved to the bathroom],[...]]
    2. fact_positions: A list, [0, 1, 2 ... #facts], the index is unique in a task (a segment split by question).
    3. task_idx： This indicates the type of this specific task, i.e. 0 to 19
    4. q: A list containing the vocabulary indexes of words of the question
    5. a: An integer, the vocabulary index of the answer.
    '''
    def encode_data(self, data):
        # see more details in get_vocab() and class Vocab
        vocab = self.get_vocab(data.train)

        def encode(tasks):
            encoded_tasks = []
            # each 'tasks'(train/val/test) has 20 task, every task corresponds to a file such as "qa1_train.txt"
            for task_idx, task in enumerate(tasks):
                encoded_questions = []
                for q in task: # the name of "task" is misleading, actually "task" here is One type of tasks
                    qfacts = q['facts'][-self.num_facts:]  # only use the last 20 facts to create graph
                    eq = [vocab[w] for w in q['q'].split(' ')]  # the indexes of words of the question
                    ea = vocab[q['a']]  # the index of the answer (only has one word)
                    # similarly, map the facts into list of indexes
                    efacts = [[vocab[w] for w in f.split(' ')] for f in qfacts]
                    efp = [i for i in range(len(qfacts))]  # 0, 1, 2 ... [the number of facts of this task]

                    encoded_questions.append({
                        'q': eq,
                        'a': ea,
                        'facts': efacts,
                        'fact_positions': efp,
                        'task_idx': task_idx
                    })
                encoded_tasks.append(encoded_questions)
            return encoded_tasks

        train = encode(data.train) # data.train corresponds to 20 files, i.e. qa*_train.txt
        valid = encode(data.valid) # data.valid corresponds to 20 files, i.e. qa*_valid.txt
        test = encode(data.test) # data.test corresponds to 20 files, i.e. qa*_test.txt

        return train, valid, test, vocab

    def random_batch(self, tasks):
        # since the tasks is a list of tasks, so use random.choice() twice
        # the first time, randomly choose a type of task(20 tasks in total)
        # the second time, randomly choose a specific task (which corresponds to a specific question)
        # the resulting list contains 160(batch_size) task randomly chosen from different type of tasks
        return [random.choice(random.choice(tasks)) for i in range(self.batch_size)]

    def save(self, name):
        self.saver.save(self.session, name)

    def load(self, name):
        self.saver.restore(self.session, name)

    def train_batch(self):
        _, _loss, _logits, _answers, _indices, _summaries, _step, _train_qsize = self.session.run(
            [self.train_step, self.loss, self.out, self.answers, self.task_indices, self.summaries, self.global_step,
             self.train_qsize_op], {self.is_training_ph: True})
        if _step % 1000 == 0:
            self._eval(self.train_writer, _answers, _indices, _logits, _summaries, _step)

        return _loss

    def val_batch(self):
        _loss, _logits, _answers, _indices, _summaries, _step = self.session.run(
            [self.loss, self.out, self.answers, self.task_indices, self.summaries, self.global_step],
            {self.is_training_ph: False})
        self._eval(self.test_writer, _answers, _indices, _logits, _summaries, _step)
        return _loss

    def test_batches(self):
        batches = []
        for task_idx, tasks in enumerate(self.test):
            print(task_idx)
            for i in range(0, len(tasks), self.batch_size):
                batch = tasks[i:i + self.batch_size]

                batch = self.encode_batch(batch, False)
                feed_dict = self.get_feed_dict(batch)
                feed_dict[self.is_training_ph] = False
                batches.append(self.session.run([self.out, self.answers, self.task_indices], feed_dict))

        return batches

    def get_batch(self, is_training):
        batch = self.random_batch(self.train if is_training else self.valid)

        return self.encode_batch(batch, is_training)

    '''
    An example of length of every list:
    batch: 160  (160 tasks, see encode_data() for the content of task)
    facts: 1204
    fact_positions: 1204
    fact_segments: 1204
    questions: 160
    answers: 160
    task_indices: 160
    edge_indices: 12734
    edge_segments: 12734
    '''
    def encode_batch(self, batch, is_training):
        facts = []
        questions = []
        answers = []
        fact_segments = []
        edge_indices = []
        edge_segments = []
        fact_positions = []
        task_indices = []
        offset = 0
        keep_prob = self.edge_keep_prob if is_training else 1.0
        for i, task in enumerate(batch):
            n_facts = len(task['facts'])
            facts += task['facts']
            fact_positions += task['fact_positions']
            questions.append(task['q'])
            answers.append(task['a'])
            # fact_segments is used with tf.gather() to match the corresponding question to each fact(Node)
            fact_segments += [i] * n_facts
            # edge_segments is used with tf.gather() to match the corresponding question to each edge
            # , #Edge=#Node**2 means that the graph contains self-loop, i.e. there exists edge like [Node_x,Node_x]
            edge_segments += [i] * n_facts ** 2
            task_indices.append(task['task_idx'])

            # offset is used to make index of every Node is unique in a batch
            # and then the index of edge (i.e.[index_Node_i, index_Node_j]) is also unique
            edge_indices.extend([[i + offset, j + offset] for i in range(n_facts) for j in range(n_facts)])
            offset += n_facts
        # save the length of every fact into "f_seq_length", and then pad the facts
        f_seq_length = [len(f) for f in facts]
        facts = pad_sequences(facts, padding='post')
        # save the length of every question into "q_seq_length", and then pad the questions
        q_seq_length = [len(q) for q in questions]
        questions = pad_sequences(questions, padding='post')

        dtype = np.uint32
        # transform the list to ndarray and return the result
        return np.array(facts, dtype), \
               np.array(fact_positions, dtype), \
               np.array(f_seq_length, dtype), \
               np.array(questions, dtype), \
               np.array(q_seq_length, dtype), \
               np.array(answers, dtype), \
               np.array(fact_segments, dtype), \
               np.array(edge_indices, dtype), \
               np.array(edge_segments, dtype), \
               np.array(task_indices, dtype), \
               keep_prob

    def _eval(self, writer, task_answers, task_indices, logits, summaries, train_step):
        writer.add_summary(summaries, train_step)

        for step in range(self.n_steps):
            argmaxes = np.argmax(logits[step], axis=1).tolist()
            correct = [0] * 20
            wrong = [0] * 20
            for actual, task_answer, task_idx in zip(argmaxes, task_answers, task_indices):
                if task_answer == actual:
                    correct[task_idx] += 1
                else:
                    wrong[task_idx] += 1

            for i, (c, w) in enumerate(zip(correct, wrong)):
                if c != 0 or w != 0:
                    task_acc = tf.Summary(
                        value=[tf.Summary.Value(tag="steps/%d/tasks/%s" % (step, i + 1), simple_value=c / (c + w))])
                    writer.add_summary(task_acc, train_step)

            overall_acc = tf.Summary(value=[tf.Summary.Value(tag="steps/%d/tasks/avg" % step,
                                                             simple_value=sum(correct) / (sum(correct) + sum(wrong)))])
            writer.add_summary(overall_acc, train_step)


class Vocab:
    """
    Acts as a dict except it returns the key for unk_token, in case a key is not present in the backing vocabulary dict.
    """

    def __init__(self, vocab, unk_token):
        vocab[unk_token] = len(vocab)  # append the word 'UNK' to the vocabulary
        self._vocab = vocab
        self._unk_token = unk_token

    def size(self):
        return len(self._vocab)

    # override __getitem__ so we can use word as index (e.g. vocab['good']) to retrieve its index in the vocabulary
    def __getitem__(self, item):
        try:
            return self._vocab[item]
        except KeyError:
            # if use a unknown word as index, return the index of 'UNK', i.e. the len of the vocabulary
            return self._vocab[self._unk_token]
