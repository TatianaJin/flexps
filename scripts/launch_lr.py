#!/usr/bin/env python

import sys
from launch_utils import launch_util

dataset_params =  {
    'a9': {
        "input" : "hdfs:///datasets/classification/a9",
        "cardinality": 32561,
        "num_dims" : 123,
    },
    'url': {
        "input" : "hdfs:///datasets/classification/url_combined",
        "cardinality": 2396130,
        "num_dims" : 3231961,
    },
    'webspam': {
        "input" : "hdfs:///datasets/classification/webspam",
        "cardinality": 350000,
        "num_dims" : 16609143,
    },
    'kdd': {
        "input" : "hdfs:///datasets/classification/kdd12",
        "cardinality": 149639105,
        "num_dims" : 54686452,
    },
    'rcv': {
        "input" : "hdfs:///datasets/classification/rcv1_train.binary",
        "cardinality": 20242,
        "num_dims" : 47236,
    }
}

hostfile = "machinefiles/local"
#hostfile = "machinefiles/5node"
dataset='rcv'
num_epoches = 5

num_iters=dataset_params[dataset]['cardinality'] * 2;

if len(sys.argv) is 2 and sys.argv[1] == 'report':
    progfile = "build/LRLossReport"
    additional_params= {
        "max_version": num_iters * num_epoches,
        "model_input": "/home/tati/svrg_a9_online_a00005_model/",
    }
else:
    progfile = "build/LinearClassifier"
    additional_params= {
        "num_epoches" : num_epoches,
        "num_iters" : num_iters,
        "kModelType" : "ASP",  # svrg_webspam_(a)sync
        #"kModelType" : "SSP",  # {ASP/SSP/BSP/SparseSSP}
        "kStaleness" : 0,
        #"optimizer": "sgd", # {sgd|svrg}
        "optimizer": "svrg", # {sgd|svrg}
        "batch_size" : 1,
        "alpha" : 0.00001, # learning rate
        #"alpha" : 0.001, # learning rate
        #"async": False, # {True|False}
        "async": True, # svrg_webspam_async
    }

params = {
    "hdfs_namenode" : "proj10",
    "hdfs_namenode_port" : 9000,
    "num_workers_per_node" : 20,
    "num_servers_per_node" : 1,

    "kStorageType" : "Vector",  # {Vector/Map}
    "trainer": "logistic", #{logistic|linear}
    "regularizer": "l2", # {none|l1|l2|elastic_net}
    #"regularizer": "none", # {none|l1|l2|elastic_net}
    #"eta1": 0, # l1 regularization factor
    #"eta2": 0, # l2 regularization factor
    "eta1": 0.001, # l1 regularization factor
    "eta2": 0.00001, # l2 regularization factor

    "report_interval": num_iters / 4,
}

params.update(additional_params)
params.update(dataset_params[dataset])

env_params = (
  "GLOG_logtostderr=true "
  "GLOG_v=-1 "
  "GLOG_minloglevel=0 "
  # this is to enable hdfs short-circuit read (disable the warning info)
  # change this path accordingly when we use other cluster
  # the current setting is for proj5-10
  "LIBHDFS3_CONF=/data/opt/course/hadoop/etc/hadoop/hdfs-site.xml"
  )

launch_util(progfile, hostfile, env_params, params, sys.argv)
