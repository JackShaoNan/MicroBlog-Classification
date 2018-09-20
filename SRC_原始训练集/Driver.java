package Final;

//import Work3.EdgeConstruct;
//import Work3.TriangleCount;

public class Driver {
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO 合并多个MapReduce任务
		//修改为只需要 输入 输出文件的形式（args[0] , args[1])
		String forIndex[] = {args[0], args[1],args[2] + "/Task1/inter"};
		Index.main(forIndex);
		String forSort[] = {args[2]+"/Task1/inter"/*args[0] */, args[2]+"/Task1/final"}; 
		// /Task1/final 中为全局特征词表
		Sort.main(forSort);
		String forFeatureVector[] = {args[2]+"/Task1/final/part-r-00000",args[1],args[2]+"/Task2/inter"};
		// /Task2/inter 中每个特征词对应的在各个文件中的TF-IDF值
		FeatureVector.main(forFeatureVector);
		String forDocumentFeatureVector[] ={args[2]+"/Task2/inter",args[2]+"/Task2/final"};
		//  /Task2/final 中为每个文件对应的10个特征词
		DocumentFeatureVector.main(forDocumentFeatureVector);
		String forClassFeatureVector[] ={args[2]+"/Task1/final/part-r-00000",args[2]+"/Task2/final",args[2]+"/Task3/inter"};
		// /Task3/inter 中为每个类对应的其中出现的所有特征词及其频数
		ClassFeatureVector.main(forClassFeatureVector);
		String forNaiveBayes[] = {args[2]+"/Task3/inter/part-r-00000",args[3],args[2]+"/Task3/final"};
		// /Task3/final 中为分类器的结果
		NaiveBayes.main(forNaiveBayes);
		String forPrecision[] = {args[2]+"/Task3/final", args[2]+"/Task4"};
		// /Task4 中为邮件分类的精度
		Precision.main(forPrecision);
	}
}
