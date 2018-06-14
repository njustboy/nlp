package com.ppp.dataminer.nlp.docclassify.vec;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ppp.dataminer.nlp.doc2vec.data.WordPair;

import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

/**
 * 使用主题模型分布向量作为特征建立分类模型
 * 
 * 利用LDA主题模型生成p(w|z) p(z|w)=p(w|z)*p(z)/p(w)
 * 
 * @author zhangwei
 *
 */
public class TopicModelVecClassifyApply {
	 // weka数据格式
	private static Instances instances;
	// arff文件读取工具
	private static ArffLoader arffLoader;
	// 分类器
	private static Classifier classify;
	// 返回分类个数
	private static int classifyNum = 3;

	private static Map<String, Integer> attributeMap = new HashMap<String, Integer>();

	static {
		File resumeStructFile = new File("model/topicModelVec.arff");
		arffLoader = new ArffLoader();
		try {
			// 读取数据表头文件
			arffLoader.setFile(resumeStructFile);
			// 获取weka的数据格式
			instances = arffLoader.getStructure();

			classify = (Classifier) SerializationHelper.read("model/topicmodelibk1.model");

			BufferedReader br = new BufferedReader(
					new InputStreamReader(new FileInputStream(new File("model/name")), "utf8"));

			String line = null;
			while ((line = br.readLine()) != null) {
				String[] segments = line.split("-->");
				attributeMap.put(segments[0].replace("_", "/"), Integer.parseInt(segments[1].trim()));
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static List<WordPair> docClassify(String position, String content) {
		double[] vec = genDocVec(position, content);
		return docClassify(vec);
	}

	private static List<WordPair> docClassify(double[] vec) {
		List<WordPair> classifyList = new ArrayList<WordPair>();
		Instance inst = new DenseInstance(301);
		inst.setDataset(instances);
		// 把数据放入instance对象中
		inst.setValue(300, "计算机软件");
		for (int i = 0; i < 300; i++) {
			inst.setValue(i, vec[i]);
		}
		instances.setClassIndex(300);
		try {
			// 各分类的分值
			double[] distributionForInstance = classify.distributionForInstance(inst);
			List<WordPair> wps = new ArrayList<WordPair>();
			for (int i = 0; i < distributionForInstance.length; i++) {
				wps.add(new WordPair(indexToName(i), distributionForInstance[i]));
			}
			Collections.sort(wps);
			int maxIndex = wps.size() > classifyNum ? classifyNum : wps.size();
			for (int i = 0; i < maxIndex; i++) {
				classifyList.add(wps.get(i));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return classifyList;
	}

	private static double[] genDocVec(String title,String content){
		double[] vec = TopicModelApply.convertContent2vec(title+content);
		return vec;
	}

	/**
	 * 分类名称和索引到映射关系，需要和weka数据结构保持一致
	 * 
	 * @param index
	 * @return
	 */
	private static String indexToName(int index) {
		String classity = "";
		switch (index) {
		case 0:
			classity = "翻译";
			break;
		case 1:
			classity = "销售行政及商务";
			break;
		case 2:
			classity = "服装/纺织/皮革";
			break;
		case 3:
			classity = "物流/仓储";
			break;
		case 4:
			classity = "公关/媒介";
			break;
		case 5:
			classity = "互联网/电子商务/网游";
			break;
		case 6:
			classity = "家政保洁";
			break;
		case 7:
			classity = "餐饮服务";
			break;
		case 8:
			classity = "物业管理";
			break;
		case 9:
			classity = "市场/营销";
			break;
		case 10:
			classity = "化工";
			break;
		case 11:
			classity = "金融/证券/期货/投资";
			break;
		case 12:
			classity = "质量安全";
			break;
		case 13:
			classity = "培训";
			break;
		case 14:
			classity = "保险";
			break;
		case 15:
			classity = "咨询/顾问";
			break;
		case 16:
			classity = "通信技术开发及应用";
			break;
		case 17:
			classity = "客服及支持";
			break;
		case 18:
			classity = "高级管理";
			break;
		case 19:
			classity = "银行";
			break;
		case 20:
			classity = "IT-品管、技术支持及其它";
			break;
		case 21:
			classity = "销售人员";
			break;
		case 22:
			classity = "机械机床";
			break;
		case 23:
			classity = "影视/媒体";
			break;
		case 24:
			classity = "人力资源";
			break;
		case 25:
			classity = "酒店旅游";
			break;
		case 26:
			classity = "汽车销售与服务";
			break;
		case 27:
			classity = "印刷包装";
			break;
		case 28:
			classity = "建筑工程与装潢";
			break;
		case 29:
			classity = "律师/法务/合规";
			break;
		case 30:
			classity = "IT-管理";
			break;
		case 31:
			classity = "医院/医疗/护理";
			break;
		case 32:
			classity = "计算机硬件";
			break;
		case 33:
			classity = "广告";
			break;
		case 34:
			classity = "公务员";
			break;
		case 35:
			classity = "汽车制造";
			break;
		case 36:
			classity = "百货零售";
			break;
		case 37:
			classity = "工程/机械/能源";
			break;
		case 38:
			classity = "运动健身";
			break;
		case 39:
			classity = "财务/审计/税务";
			break;
		case 40:
			classity = "行政/后勤";
			break;
		case 41:
			classity = "采购";
			break;
		case 42:
			classity = "房地产销售与中介";
			break;
		case 43:
			classity = "编辑出版";
			break;
		case 44:
			classity = "贸易";
			break;
		case 45:
			classity = "交通运输服务";
			break;
		case 46:
			classity = "环保";
			break;
		case 47:
			classity = "计算机软件";
			break;
		case 48:
			classity = "生产/营运";
			break;
		case 49:
			classity = "休闲娱乐";
			break;
		case 50:
			classity = "电子/电器/半导体/仪器仪表";
			break;
		case 51:
			classity = "科研";
			break;
		case 52:
			classity = "农/林/牧/渔";
			break;
		case 53:
			classity = "美容保健";
			break;
		case 54:
			classity = "房地产开发";
			break;
		case 55:
			classity = "教师";
			break;
		case 56:
			classity = "艺术/设计";
			break;
		case 57:
			classity = "生物/制药/医疗器械";
			break;
		case 58:
			classity = "网店淘宝";
			break;
		case 59:
			classity = "技工普工";
			break;
		default:
			break;
		}
		return classity;
	}
}
