import unittest

from .embedrank_s2v import Sent2VecEmbedRank


class EmbedRankS2VTest(unittest.TestCase):

    def testFromFile(self):
        input_file = '/opt/algo_nfs/kdd_luozhouyang/embedrank/part-00000.sent2vec.txt'
        model_path = '/opt/algo_nfs/kdd_luozhouyang/embedrank/model.sent2vec.100d.bin.bin'
        model = Sent2VecEmbedRank(model_path)

        with open(input_file, mode='rt', encoding='utf8') as fin:
            for i in range(1000):
                line = fin.readline()
                if not line:
                    break
                line = line.strip('\n').strip()
                keywords = model.extract_keyword(line)
                print(sorted(keywords, key=lambda x: x[1], reverse=True))

    def testDocs(self):
        model_path = '/opt/algo_nfs/kdd_luozhouyang/embedrank/model.sent2vec.100d.bin.bin'
        model = Sent2VecEmbedRank(model_path)

        docs = [
            "前端开发",
            "java初级工程师(福田区)",
            "熟悉java开发，熟悉分布式，熟悉前端的react、vue框架。",
            "技能要求:1、计算机相关专业,本科及以上学历,2年以上java开发经验; 2、精通java,熟悉j2ee开发,熟练使用oracle/mssqlserver2005等数据库; 3、熟悉struts、spring、hiberate等框架; 4、熟悉常用的前端框架:ajax技术,熟悉jquery、extjs更佳; 5、有过金融/物流方面经验者优先。",
            "职责描述: 1.负责项目部分模块的设计开发工作; 2.指导协助同事解决日常工作中的问题 3.对系统存在的问题进行跟踪和定位并及时解决; 4.严格执行工作计划,主动汇报并高效完成任务保证部门及个人工作目标实现; 5.注重学习和积累,追求卓越; 任职要求: 1.计算机及相关专业本科,4年及以上工作经验; 2. 熟练使用java 编程,有良好的编码习惯;  •3.熟悉大数据处理相关产品架构和技术(如storm/hadoop/hive/hbase/spark/kafka/flume/zookeeper/redis等); 4.使用过storm/spark streaming优先; 5. 熟练使用linux开发环境和命令; 6.熟悉主流的数据库技术(oracle、mysql等); 7.具备良好的学习能力,分析解决问题和沟通表达能力",
        ]
        for d in docs:
            print(sorted(model.extract_keyword(d), key=lambda x: x[1], reverse=True))
            print('=' * 100)


if __name__ == "__main__":
    unittest.main()
