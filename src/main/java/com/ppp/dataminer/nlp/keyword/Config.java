package com.ppp.dataminer.nlp.keyword;

public class Config {
   // 如果一个词属于职位名，则需要增加的权重
   public static double JOBTITLE_WEIGHT = 2;
   // 词语在文本中的位置对词重要性的权重，0表示与位置无关
   public static double POSITION_WEIGHT = 0.5;
   // TF-IDF公式中IDF的权重
   public static double IDF_WEIGHT = 1.0;
   // 计算两个关键词之间关系的公式中的TF-IDF值的权重
   public static double TFIDF_WEIGHT = 1.0;
   // 默认的匹配正确率
   public static double MATCH_CORRECT_RATE = 1.0;
   // 计算两个关键词之间关系的公式中的关键词相似度的权重
   public static double DISTANCE_WEIGHT = 1.0;
   // 关键词在整个文本中所占比例
   public static double KEYWORD_PERCENT = 0.3;
   // 关键词个数上限
   public static int KEYWORD_MAXCOUNT = 20;
   // JOBIDF
   public static double JOBIDF_MAX = 12.0;
   // WORKIDF
   public static double WORKIDF_MAX = 17.0;
   // 关键词个数下限
   public static int KEYWORD_MINCOUNT = 3;
   // 关键词长度的权重
   public static double KEYWORD_LENGTH_WEIGHT = 1;
   // TFIDF关键字组合时所占的比例
   public static double TFIDF_PERCENT = 0.2;
   // 词性关键字组合时所占的比例
   public static double CIXING_PERCENT = 0.2;
   // 主题支持度权重
   public static double TOPIC_PERCENT = 0.3;
   // 语义支持度权重
   public static double SEMANTIC_PERCENT = 0.3;

   public static double UPDATE_WHEN_NULL = 0.1;
}
