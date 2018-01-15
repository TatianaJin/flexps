#!/usr/bin/env python

import sys
from launch_utils import launch_util

hostfile = "machinefiles/5node"
progfile = "build/LinearClassifier"

params = {
    "hdfs_namenode" : "proj10",
    "hdfs_namenode_port" : 9000,
    "num_workers_per_node" : 20,
    "num_servers_per_node" : 1,

    "input" : "hdfs:///datasets/classification/url_combined",
    "cardinality": 2396130,
    "num_dims" : 3231961,
    "num_iters" : 47922,

    #"input" : "hdfs:///datasets/classification/webspam",
    #"cardinality": 350000,
    #"num_dims" : 16609143,
    #"num_iters" : 7000,

    #"input" : "hdfs:///datasets/classification/kdd12",
    #"cardinality": 149639105,
    #"num_dims" : 54686452,
    #"num_iters" : 1496391*2,

    "kStorageType" : "Vector",  # {Vector/Map}
    "kModelType" : "ASP",  # {ASP/SSP/BSP/SparseSSP}
    "kStaleness" : 0,
    "trainer": "logistic", #{logistic|linear}
    "optimizer": "svrg", # {sgd|svrg}
    "batch_size" : 1,
    "report_interval": 0,
    "alpha" : 0.1, # learning rate
    "regularizer": "elastic_net", # {none|l1|l2|elastic_net}
    "eta1": 0.001, # l1 regularization factor
    "eta2": 0.001, # l2 regularization factor
    "async": True, # {True|False}
}

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
