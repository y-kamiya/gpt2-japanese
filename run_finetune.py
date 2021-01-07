# coding=utf-8

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from copy import copy
from tensorflow.contrib.training import HParams
from encode_bpe import BPEEncoder_ja
import model
import csv

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

parser = argparse.ArgumentParser(
    description='Pretraining GPT2-JA on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input npz file')
parser.add_argument('--base_model', type=str, default='gpt2ja-small', help='a path to a model file')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--optim', type=str, default='adam', help='"adam", "adagrad", or "sgd" to use optimizer')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=5e-5, help='Learning rate for optimizer')
parser.add_argument('--warmup_steps', metavar='WR', type=int, default=0, help='Learning rate warming up steps')

parser.add_argument('--run_name', type=str, default='gpt2ja_finetune', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')

parser.add_argument('--gpu', default='0', help='visible gpu number.')

parser.add_argument('--n_sentence_labels', type=int, default=0, help='label num for sentence classification')
parser.add_argument('--epochs', type=int, default=100)

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

with open('ja-bpe.txt') as f:
    bpe = f.read().split('\n')

with open('emoji.json') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)

LABEL_INDEX_MAP = {
    'anger': 0,
    'disgust': 1,
    'joy': 2,
    'sadness': 3,
    'surprise': 4,
}

TOKEN_ID_EOT = enc.encode('<|endoftext|>')[0]

class Dataset():
    def __init__(self, args, hparams, filename):
        self.args = args
        self.hparams = hparams
        self.filename = filename

        self.global_chunks, self.global_label_ids = self.__load_data()
        self.global_chunk_index = np.random.permutation(len(self.global_chunks))
        self.global_chunk_step = 0

    def __load_data(self):
        input_ids = []
        label_ids = []
        input_path = os.path.join(self.args.dataset, self.filename)
        with open(input_path, 'r') as fp:
            reader = csv.reader(fp, delimiter='\t')
            for text, label in reader:
                input_ids.append(enc.encode(text)[:self.hparams.n_ctx])
                label_ids.append(LABEL_INDEX_MAP[label])

        return input_ids, label_ids


def evaluate(args, hparams):
    input_ids, label_ids = load_data_for_classification(args, hparams, 'eval.txt')


def main():
    args = parser.parse_args()

    if 'small' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 768,
          "n_head": 12,
          "n_layer": 12
        })
    elif 'medium' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1024,
          "n_head": 16,
          "n_layer": 24
        })
    elif 'large' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1280,
          "n_head": 20,
          "n_layer": 36
        })
    else:
        raise ValueError('invalid model name.')

    config = tf.ConfigProto()
    if int(args.gpu) >= 0:
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = args.gpu
    with tf.Session(config=config,graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [None, None])
        labels = tf.placeholder(tf.int32, [None])
        eos_indices = tf.placeholder(tf.int32, [None, 2])
        output = model.model(hparams=hparams, X=context, past=None , reuse=tf.AUTO_REUSE)

        saver = tf.train.Saver(
            var_list=tf.trainable_variables(),
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.latest_checkpoint(args.base_model)
        saver.restore(sess, ckpt)
        print('Loading checkpoint', ckpt)
        # saver = tf.train.Saver()
        # ckpt = tf.train.latest_checkpoint(args.base_model)
        # saver.restore(sess, ckpt)

        logits = model.sentence_classification_head(hparams, output['h_norm'], eos_indices, args.n_sentence_labels)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))

        preds = tf.math.argmax(logits, 1)
        accuracy_op = tf.reduce_mean(tf.cast(preds == labels, tf.float32))

        global_step = tf.Variable(0, trainable=False)
        if args.warmup_steps > 0:
            learning_rate = tf.compat.v1.train.polynomial_decay(
                    learning_rate=1e-10,
                    end_learning_rate=args.learning_rate,
                    global_step=global_step,
                    decay_steps=args.warmup_steps
                )
        else:
            learning_rate = args.learning_rate

        if args.optim=='adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-7)
        elif args.optim=='adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif args.optim=='sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError('invalid optimizer name.')

        train_vars = tf.trainable_variables()
        opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)

        summaries_train = tf.summary.scalar('train/loss', loss)
        summaries_eval = tf.summary.scalar('eval/loss', loss)
        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)

        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        sess.run(tf.variables_initializer(not_initialized_vars))

        dataset_train = Dataset(args, hparams, 'train.txt')
        dataset_eval = Dataset(args, hparams, 'eval.txt')

        def sample_feature(dataset):
            p_input_ids = []
            p_label_ids = []

            for b in range(args.batch_size): # FULL-SENTENCES
                idx = dataset.global_chunk_index[dataset.global_chunk_step]
                dataset.global_chunk_step += 1
                if dataset.global_chunk_step >= len(dataset.global_chunk_index):
                    dataset.global_chunk_step = 0
                    dataset.global_chunk_index = np.random.permutation(len(dataset.global_chunks))

                p_input_ids.append(copy(dataset.global_chunks[idx]))
                p_label_ids.append(copy(dataset.global_label_ids[idx]))

            if args.n_sentence_labels == 0:
                return {context:p_input_ids}

            sentence_lengths = []
            max_length = max([len(ids) for ids in p_input_ids])
            for i in range(len(p_input_ids)):
                length = len(p_input_ids[i])
                n_diff = max_length - length + 1
                p_input_ids[i].extend([TOKEN_ID_EOT] * n_diff)
                sentence_lengths.append(length)

            p_eos_indices = list(zip(range(len(sentence_lengths)), sentence_lengths))
            # print('nnnnnnnnnnnnnnnnnnnnnnnnnn')
            # print(p_input_ids)
            # print(p_label_ids)
            # print(p_eos_indices)
            return {context:p_input_ids, labels:p_label_ids, eos_indices:p_eos_indices}


        print('Training...')

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        maketree(os.path.join(CHECKPOINT_DIR, args.run_name))

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if counter % args.save_every == 0:
                    (v_accuracy, v_loss, v_summary) = sess.run(
                        (accuracy_op, loss, summaries_eval),
                        feed_dict=sample_feature(dataset_eval))
                    print(
                        '[eval][{counter} | {time:2.2f}] loss={loss:2.2f} acc={acc:2.2f}'
                        .format(
                            counter=counter,
                            time=time.time() - start_time,
                            loss=v_loss,
                            acc=v_accuracy))
                    summary_log.add_summary(v_summary, counter)
                    save()

                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summaries_train),
                    feed_dict=sample_feature(dataset_train))

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter = counter+1
                if args.warmup_steps > 0:
                    global_step = global_step+1

        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    main()
