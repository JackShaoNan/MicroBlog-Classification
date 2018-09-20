package Final;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

public class NaiveBayes {
	public static class NBMap extends Mapper<LongWritable, Text, Text, Text> {
		
		protected void map(LongWritable key, Text value, Context context)
				throws java.io.IOException, InterruptedException {
			String filePath = ((FileSplit)context.getInputSplit()).getPath().toString(); //获取文件路径
			String[] splits;
			splits = filePath.split("/");
			filePath = splits[splits.length - 2]+"@"+splits[splits.length - 1];//类别名@文件名
			
			Analyzer analyzer = new StandardAnalyzer();//分词，注意没有使用停词表，暂时默认为分词无差别，有差别的也不是什么好词
			TokenStream tokens = analyzer.tokenStream("", value.toString());
			tokens.reset();
			while (tokens.incrementToken()) {
				CharTermAttribute charTerm = tokens.getAttribute(CharTermAttribute.class);
				context.write(new Text(filePath), new Text(charTerm.toString()));//类@文件做key 词做value
			}
			analyzer.close();
		}
	}
	
	public static class NBReduce extends Reducer<Text, Text, Text, Text> {
		public Map<String, Map<String, Double>> wordToClass = new HashMap<>(); 
		public String[] classNo = {"alt.atheism", "sci.space", "rec.autos",
				"comp.graphics", "rec.motorcycles", "soc.religion.christian",
				"comp.os.ms-windows.misc", "rec.sport.baseball", "talk.politics.guns",
				"comp.sys.ibm.pc.hardware", "rec.sport.hockey", "talk.politics.mideast",
				"comp.sys.mac.hardware", "sci.crypt", "talk.politics.misc",
				"comp.windows.x", "sci.electronics", "talk.religion.misc",
				"misc.forsale", "sci.med"}; //预设20个类及其对应编号
		
		protected void setup(Context context) throws IOException {
			Configuration configuration = context.getConfiguration();
			// 设置全局表，key是特征词，value是一张map，对应着改特征词在每个类中TF-IDF值之和
			Path path = new Path(configuration.get("wordToClass"));
			BufferedReader bufferedReader = new BufferedReader(new FileReader(path.toString()));
			String line;
			while((line = bufferedReader.readLine()) != null){
				String[] strings = line.split("\t");// 读入的一行是“类		词#TF-IDF值“
				String[] post = strings[1].split(" ");
				for(String string: post){
					String[] temp = string.split("#");
					if(wordToClass.containsKey(temp[0])){
						Map<String, Double> tempMap = wordToClass.get(temp[0]);
						if(tempMap.containsKey(strings[0])){//strings[0]为类名,temp1为TF-IDF值
							double val = tempMap.get(strings[0])+Double.parseDouble(temp[1]);
							tempMap.remove(strings[0]);
							tempMap.put(strings[0], val);//更新TF-IDF的总值
						}
						else {//找不到该类
							tempMap.put(strings[0], Double.parseDouble(temp[1]));
						}
					}
					else {
						Map<String, Double> tempMap = new HashMap<>();
						tempMap.put(strings[0], Double.parseDouble(temp[1]));//类->TF-IDF总值
						wordToClass.put(temp[0], tempMap);//词 -> 类信息
					}
				}
			}
			bufferedReader.close();
		}
		
		protected void reduce(Text word, Iterable<Text> values, Context context)
				throws java.io.IOException, InterruptedException {
			double[] nbResult = new double[20];
			
			// 迭代器只能遍历一次，由于需要遍历20次，所以干脆转换为一个数组
			ArrayList<String> tokens = new ArrayList<>();
			for(Text v: values){
				tokens.add(v.toString());
			}
			
			// 计算邮件对应每个类的贝叶斯值，由于特征词可能在一些类中没有出现，所以加入拉普拉斯校正
			for(int i=0; i<20; i++){
				String className = classNo[i];
				nbResult[i]=1.0;
				for(String token : tokens){
					if(wordToClass.containsKey(token)){
						Map<String, Double> temp = wordToClass.get(token); //对应的每个类下的TF-IDF值之和
						double sum = 0.0;
						for(Entry<String, Double> entry : temp.entrySet()){
							sum+=entry.getValue();
						}//计算词被选为特征词的总数
						if(temp.containsKey(className)){ //该特征词在当前类（className)中出现过
							nbResult[i]*=(1.0+(double)temp.get(className))/(20.0+(double)sum);
							//贝叶斯加拉普拉斯校正
						}
						else {
							nbResult[i]*=1.0/((20.0+(double)sum)*0.5);//贝叶斯加拉普拉斯校正
						}
					}
				}
			}
			
			int max = 0;
			for(int i=1; i<20; i++){
				if(nbResult[max]<nbResult[i]){
					max=i;//选取最大的
				}
			}
			context.write(word, new Text(classNo[max])); //此处word为输入的key:  类名(精确值)@文件名
			tokens.clear();
		}
	}
	
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		
		String[] wTC = args[0].split("/");
		conf.set("wordToClass", wTC[wTC.length-1]);
		
		Job job3_5 = new Job(conf, "Task3.5");
		job3_5.setJarByClass(FeatureVector.class);
		
		job3_5.setMapperClass(NBMap.class);
		job3_5.setReducerClass(NBReduce.class);
		
		job3_5.setMapOutputKeyClass(Text.class);
		job3_5.setMapOutputValueClass(Text.class);
		job3_5.setOutputKeyClass(Text.class);
		job3_5.setOutputValueClass(Text.class);
		
		job3_5.addCacheFile(new Path(args[0]).toUri());
		
		// the parameter.
		FileInputFormat.addInputPath(job3_5, new Path(args[1]));
		FileInputFormat.setInputDirRecursive(job3_5, true);
		FileOutputFormat.setOutputPath(job3_5, new Path(args[2]));
		// wait and print the process, return true when finish successfully.
		//System.exit(job.waitForCompletion(true) ? 0 : 1);
		 job3_5.waitForCompletion(true);
	}
	
}
