import json
import os
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
from tensorflow.contrib.training import HParams
from sampling import sample_sequence
from encode_bpe import BPEEncoder_ja

END_OF_TEXT = '<|endoftext|>'

parser = argparse.ArgumentParser()
parser.add_argument('output_file', type=str)
parser.add_argument('--model', type=str, default='gpt2ja-medium')
parser.add_argument('--context', type=str, default=END_OF_TEXT)
parser.add_argument('--context_file', type=str, default=None)
parser.add_argument('--num_generate', type=int, default=5)
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--top_p', type=float, default=0)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--max_length', type=int, default=500)
args = parser.parse_args()

with open('ja-bpe.txt') as f:
    bpe = f.read().split('\n')

with open('emoji.json') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)

if 'small' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12
    })
elif 'medium' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 1024,
      "n_head": 16,
      "n_layer": 24
    })
elif 'large' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 1280,
      "n_head": 20,
      "n_layer": 36
    })
else:
    raise ValueError('invalid model name.')

length=hparams.n_ctx // 2
temperature=args.temperature
top_k=args.top_k
top_p=args.top_p

def generate_one(sess, output, pre_text):
    generated = ''
    while True:
        context_tokens = enc.encode(pre_text)
        if len(context_tokens) > length:
            context_tokens = context_tokens[-length:]
        out = sess.run(output, feed_dict={
            context: [context_tokens]
        })[:,len(context_tokens):]
        swd = enc.decode(out[0])
        last = False
        if END_OF_TEXT in swd:
            swd = swd.split(END_OF_TEXT)[0]
            last = True
        if len(swd) > 0:
            generated += swd

        if last or len(generated) > args.max_length:
            if len(generated) > 0:
                return generated[:args.max_length]
        else:
            pre_text = generated[-256:]


config = tf.ConfigProto()
if int(args.gpu) >= 0:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.gpu

with tf.Session(config=config, graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [1, None])
    output = sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=1,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, ckpt)

    contexts = [args.context]

    if args.context_file is not None and os.path.exists(args.context_file):
        with open(args.context_file) as f:
            contexts = [line.strip() for line in f.readlines()]

    with open(args.output_file, 'w') as of:
        for pre_text in contexts:
            for i in range(args.num_generate):
                generated = generate_one(sess, output, pre_text)
                generated = generated.replace('\n', '')
                of.write(f"{pre_text}{generated}\n")
