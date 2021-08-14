# 知识蒸馏
这是一个相较于剪枝比较优秀的模型简化方式，其原理不难理解，通过训练一个老师模型（复杂模型），通过老师模型所得到的消息通过蒸馏来给予学生用来学习，从而得到一个轻量且误差相较于老师模型不大的学生模型。

当然，这种方式本身是存在很多种算法的，可能会在之后的时间里一一进行说明。

| Name     | Method                                  |                          Paper Link                          |                          Code Link                           |
| :------- | --------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Baseline | basic model with softmax loss           |                              —                               | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/train_base.py) |
| Logits   | mimic learning via regressing logits    | [paper](http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/logits.py) |
| ST       | soft target                             |        [paper](https://arxiv.org/pdf/1503.02531.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/st.py) |
| AT       | attention transfer                      |        [paper](https://arxiv.org/pdf/1612.03928.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py) |
| Fitnet   | hints for thin deep nets                |         [paper](https://arxiv.org/pdf/1412.6550.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/fitnet.py) |
| NST      | neural selective transfer               |        [paper](https://arxiv.org/pdf/1707.01219.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/nst.py) |
| PKT      | probabilistic knowledge transfer        | [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/pkt.py) |
| FSP      | flow of solution procedure              | [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/fsp.py) |
| FT       | factor transfer                         | [paper](http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ft.py) |
| RKD      | relational knowledge distillation       |        [paper](https://arxiv.org/pdf/1904.05068.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/rkd.py) |
| AB       | activation boundary                     |        [paper](https://arxiv.org/pdf/1811.03233.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ab.py) |
| SP       | similarity preservation                 |        [paper](https://arxiv.org/pdf/1907.09682.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/sp.py) |
| Sobolev  | sobolev/jacobian matching               |        [paper](https://arxiv.org/pdf/1706.04859.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/sobolev.py) |
| BSS      | boundary supporting samples             |        [paper](https://arxiv.org/pdf/1805.05532.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/bss.py) |
| CC       | correlation congruence                  | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/cc.py) |
| LwM      | learning without memorizing             |        [paper](https://arxiv.org/pdf/1811.08051.pdf)         | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/lwm.py) |
| IRG      | instance relationship graph             | [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/irg.py) |
| VID      | variational information distillation    | [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/vid.py) |
| OFD      | overhaul of feature distillation        | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ofd.py) |
| AFD      | attention feature distillation          |      [paper](https://openreview.net/pdf?id=ryxyCeHtPB)       | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/afd.py) |
| CRD      | contrastive representation distillation |      [paper](https://openreview.net/pdf?id=SkgpBJrtvS)       | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/crd.py) |
| DML      | deep mutual learning                    | [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/dml.py) |
