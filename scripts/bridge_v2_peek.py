#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, glob, json
import tensorflow as tf

def peek_one(path):
    for raw in tf.compat.v1.io.tf_record_iterator(path):
        ex = tf.train.Example()
        ex.ParseFromString(raw)
        return sorted(ex.features.feature.keys())
    return []

def main():
    tfrec_glob = sys.argv[1] if len(sys.argv) > 1 else "/mnt/BridgeLang/data/bridgedata_v2/raw/tfds/bridge_dataset-val.tfrecord-*"
    files = sorted(glob.glob(tfrec_glob))
    assert files, f"No TFRecord files for glob: {tfrec_glob}"
    keys = peek_one(files[0])
    print("First shard:", files[0])
    print("Feature keys (first example):")
    for k in keys:
        print(" -", k)

if __name__ == "__main__":
    main()
