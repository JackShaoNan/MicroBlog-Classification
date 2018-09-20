package Final;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

public class FeatureVector {
	public static class FVMap extends Mapper<LongWritable, Text, Text, Text> {
		public Set<String> FVSet = new HashSet<>();
		
		protected void setup(Context context) throws IOException {
			Configuration configuration = context.getConfiguration();
			
			Path path = new Path(configuration.get("EigenTable"));
			BufferedReader bufferedReader = new BufferedReader(new FileReader(path.toString()));
			String line;
			while((line = bufferedReader.readLine()) != null){
				String[] strings = line.split("\t");// 读入的一行是“特征词 编号，总数“
				FVSet.add(strings[0]);
			}
			bufferedReader.close();
		}
		
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			String filePath = ((FileSplit)context.getInputSplit()).getPath().toString(); //获取文件路径
			String[] splits;
			splits = filePath.split("/");
			filePath = splits[splits.length - 2]+"@"+splits[splits.length - 1];//类别名@文件名
			Text word = new Text();
			
			Analyzer analyzer = new StandardAnalyzer();//分词，注意没有使用停词表，暂时默认为分词无差别，有差别的也不是什么好词
			TokenStream tokens = analyzer.tokenStream("", value.toString());
			tokens.reset();
			while (tokens.incrementToken()) {
				CharTermAttribute charTerm = tokens.getAttribute(CharTermAttribute.class);
				if(FVSet.contains(charTerm.toString())){//是一个特征词
					word.set(charTerm.toString() + "\t" + filePath);
					context.write(word, new Text());//词+类@文件做Outkey	同时Outvalue 为计数1
 				}
			}
			analyzer.close();
		}
	}
	
	public static class FVPartitioner extends HashPartitioner<Text, IntWritable> {
		public int getPartition(Text key, IntWritable value, int numReduceTasks) {

			String term = new String();
			term = key.toString().split("\t")[0];
			return super.getPartition(new Text(term), value, numReduceTasks);
		}
	}
	
	public static class FVCombine extends Reducer<Text, Text, Text, Text>{
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws java.io.IOException ,InterruptedException {
            String[] splits = key.toString().split("\t");
            if(splits.length != 2){
                return ;
            }
            
            long count = 0;
            for (Text v : values) {
                count++;
            }
            Text word = new Text(splits[0]);
            Text post = new Text(splits[1]+"#"+count);
            context.write(word, post);
        };
    }
	
	public static class FVReduce extends Reducer<Text, Text, Text, Text> {
		public Map<String, ArrayList<Long>> FVTable = new HashMap<>();
		
		protected void setup(Context context) throws IOException {
			Configuration configuration = context.getConfiguration();
			
			Path path = new Path(configuration.get("EigenTable"));
			BufferedReader bufferedReader = new BufferedReader(new FileReader(path.toString()));
			String line;
			while((line = bufferedReader.readLine()) != null){
				String[] strings = line.split("\t");// 读入的一行是“特征词   编号,总数“
				String[] nAndS = strings[1].split(",");
				ArrayList<Long> numAndSum = new ArrayList<>();
				numAndSum.add(Long.parseLong(nAndS[0]));//编号
				numAndSum.add(Long.parseLong(nAndS[1]));//总数
				FVTable.put(strings[0], numAndSum);
			}
			bufferedReader.close();
		}
		
		protected void reduce(Text word, Iterable<Text> values, Context context)
				throws java.io.IOException, InterruptedException {
			double N = 20000.0-2264.0;
			long no = FVTable.get(word.toString()).get(0);//编号
			double idf = log2(N/(((double)FVTable.get(word.toString()).get(1))));
			StringBuilder buffer = new StringBuilder();
			for(Text v: values) {
				String temp = v.toString();
				String[] strings = temp.split("#");
				if (strings.length != 2) {
					return;
				}
				long tf = Long.parseLong(strings[1]);//TF，同时也是词频
				double tf_idf = idf*((double)(1.0+log2((double)tf)));
				buffer.append(strings[0]+"#"+Double.toString(tf_idf)+";");
				//string[0]为类名@文件名
			}
			Text postings = new Text(buffer.toString());
			Text wordNo = new Text(Long.toString(no));
			context.write(wordNo, postings);
		}
	}
	
	public static double log2(double a) {
		return Math.log(a)/Math.log(2.0);
	}
	
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		String[] cachePath = args[0].split("/");
		conf.set("EigenTable", cachePath[cachePath.length-1]);
		Job job2 = new Job(conf, "Task2");
		job2.setJarByClass(FeatureVector.class);
		
		job2.setMapperClass(FVMap.class);
		job2.setPartitionerClass(FVPartitioner.class);
		job2.setCombinerClass(FVCombine.class);
		job2.setReducerClass(FVReduce.class);
		
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(Text.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		job2.addCacheFile(new Path(args[0]).toUri());
		
		// the parameter.
		FileInputFormat.addInputPath(job2, new Path(args[1]));
		FileInputFormat.setInputDirRecursive(job2, true);
		FileOutputFormat.setOutputPath(job2, new Path(args[2]));
		// wait and print the process, return true when finish successfully.
		//System.exit(job2.waitForCompletion(true) ? 0 : 1);
		   job2.waitForCompletion(true);
	}

}
