from __future__ import division
from __future__ import print_function
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from constructor import get_placeholder, update
from input_data import format_data
from sklearn.metrics import roc_auc_score
from model import *
from optimizer import *
from recallAtK import calculateRecallAtK
from precisionAtK import calculatePrecisionAtK
import logging  # 引入logging模块
import os.path
import time
import  pandas as pd


# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/Logs/'
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)
# 日志
logger.debug('this is a logger debug message')
logger.info('this is a logger info message')
logger.warning('this is a logger warning message')
logger.error('this is a logger error message')
logger.critical('this is a logger critical message')
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

#df需要提前排序
def calc_pk(df):
    df["actual_sum"] = df["actual"].cumsum()
    df["k"] = 1
    df["k"] = df["k"].cumsum()#截止到k 查询到的正确值
    df['p@k'] = df.apply(lambda x: x["actual_sum"] / x["k"], axis=1)
    return df
def calc_rk(df):
    df["actual_sum"] = df["actual"].cumsum()
    df["all"] = 300#异常节点的总数量
    df['r@k']=df.apply(lambda x: x["actual_sum"] / x["all"], axis=1)
    return df


#或得评价结果矩阵
#encoder_df["p@k"] = encoder_df.apply(lambda x: x["actual_sum"] / x["k"], axis=1)


def precision_AT_K(actual, predicted, k, num_anomaly):
    act_set = np.array(actual[:k])
    pred_set = np.array(predicted[:k])
    ll = act_set & pred_set
    tt = np.where(ll == 1)[0]
    prec = len(tt) / float(k)
    rec = len(tt) / float(num_anomaly)
    return round(prec, 4), round(rec, 4)

class AnomalyDetectionRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.decoder_act = settings['decoder_act']

        #添加打印属性
        self.alpha =settings['alpha']
        self.eta =settings['eta']
        self.theta =settings['theta']
        self.embed_dim=settings['embed_dim']
        self.iterations=settings['iterations']





    def erun(self, writer):
        model_str = self.model
        # load data
        feas = format_data(self.data_name)

        print("feature number: {}".format(feas['num_features']))
        # Define placeholders
        placeholders = get_placeholder()
        #num_nodes = adj.shape[0]

        num_features = feas['num_features']
        features_nonzero = feas['features_nonzero']
        num_nodes = feas['num_nodes']

        if model_str == 'Dominant':
            model = GCNModelAE(placeholders, num_features, features_nonzero)
            opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
                              labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                              preds_structure=model.structure_reconstructions,
                              labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha)

        elif model_str == 'AnomalyDAE':
            model = AnomalyDAE(placeholders, num_features, num_nodes, features_nonzero, self.decoder_act)
            opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
                               labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                               preds_structure=model.structure_reconstructions,
                               labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha,
                               eta=FLAGS.eta, theta=FLAGS.theta)
            # opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
            #                   labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
            #                   preds_structure=model.structure_reconstructions,
            #                   labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha)
        elif model_str == 'GCNModelVAE':

            model = GCNModelVAE(placeholders, num_features,num_nodes, features_nonzero,self.decoder_act)
            # opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
            #                    labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
            #                    preds_structure=model.structure_reconstructions,
            #                    labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha,
            #                    eta=FLAGS.eta, theta=FLAGS.theta)
            opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
                              labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                              preds_structure=model.structure_reconstructions,
                              labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha)
        elif model_str == 'GCNModelGAN':

            model = GCNModelGAN(placeholders, num_features,num_nodes, features_nonzero,self.decoder_act)
            # opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
            #                    labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
            #                    preds_structure=model.structure_reconstructions,
            #                    labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha,
            #                    eta=FLAGS.eta, theta=FLAGS.theta)
            opt = OptimizerGAN(preds_attribute=model.attribute_reconstructions,
                              labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                              preds_structure=model.structure_reconstructions,
                              labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha)

        # Initialize session

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()

        # Train model
        maxAUC = 0
        y_true=0
        scores =0
        for epoch in range(1, self.iteration+1):

            train_loss, loss_struc, loss_attr, rec_error = update(model, opt, sess,
                                                                feas['adj_norm'],
                                                                feas['adj_label'],
                                                                feas['features'],
                                                                placeholders, feas['adj'])

            if epoch % 1 == 0:
                y_true = [label[0] for label in feas['labels']]

                auc=0
                try:
                    scores = np.array(rec_error)
                    scores = (scores - np.min(scores)) / (
                            np.max(scores) - np.min(scores))

                    auc = roc_auc_score(y_true, scores)


                except Exception:
                    print("[ERROR] for auc calculation!!!")
                    logger.info("[ERROR] for auc calculation!!!")

                print("Epoch:", '%04d' % (epoch),
                      "AUC={:.5f}".format(round(auc,4)),
                      "train_loss={:.5f}".format(train_loss),
                      "loss_struc={:.5f}".format(loss_struc),
                      "loss_attr={:.5f}".format(loss_attr))
                logger.warning("Epoch:{}AUC={:.5f} train_loss={:.5f} loss_struc={:.5f} loss_attr={:.5f}".format(epoch,round(auc,4),train_loss,loss_struc,loss_attr))

                if round(auc,4) > maxAUC:
                    maxAUC = round(auc,4)

                writer.add_scalar('loss_total', train_loss, epoch)
                writer.add_scalar('loss_struc', loss_struc, epoch)
                writer.add_scalar('loss_attr', loss_attr, epoch)
                writer.add_scalar('auc', auc, epoch)

        print(" model:{},data_name:{},iterations:{},alpha:{},eta:{},theta{},embed_dim:{}".format(model,self.data_name,self.iterations,self.alpha,self.eta,self.theta,self.embed_dim))
        print(" MAX AUC={:.5f}".format(round(maxAUC,4)))
        print(y_true)
        print(scores)
        #输出结果文件
        node_number = num_nodes
        df = pd.DataFrame()
        a = [x for x in range(node_number)]
        # a=np.array([5,8,9])
        df['id'] = a
        df['score']=scores
        df['label']=y_true
        df.to_csv('{}_result.csv'.format(self.data_name))


        logger.warning(" MAX AUC={:.5f}".format(round(maxAUC,4)))
        logger.warning(" model:{},data_name:{},iterations:{},alpha:{},eta:{},theta{},embed_dim:{}".format(model,self.data_name,self.iterations,self.alpha,self.eta,self.theta,self.embed_dim))




