package Final;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
//import org.apache.hadoop.fs.shell.Count;
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
import org.apache.lucene.analysis.util.CharArraySet;

public class Index {
	
	public static class MyMap extends Mapper<LongWritable, Text, Text, Text> {
		public ArrayList<String> arrayList = new ArrayList<>();
		
		protected void setup(Context context) throws IOException {//加入一个全局的停词表
			Configuration configuration = context.getConfiguration();
			
			Path path = new Path(configuration.get("cache"));
			BufferedReader bufferedReader = new BufferedReader(new FileReader(path.toString()));
			String line;
			while((line = bufferedReader.readLine()) != null){
				arrayList.add(line);
			}
			bufferedReader.close();
		}
		
		protected void map(LongWritable key, Text value, Context context)
				throws java.io.IOException, InterruptedException {
			String parentPath = ((FileSplit) context.getInputSplit()).getPath().toString(); //获取文件路径
			String[] splits;
			splits = parentPath.split("/");
			String name = splits[splits.length - 1];//文件名
			parentPath = splits[splits.length - 2];//类别名
			Text word = new Text();

//			Analyzer analyzer = new StandardAnalyzer();//分词
			CharArraySet charArraySet = new CharArraySet(arrayList, true);
			Analyzer analyzer = new StandardAnalyzer(charArraySet);
			TokenStream tokens = analyzer.tokenStream("", value.toString());
			tokens.reset();
			while (tokens.incrementToken()) {
				CharTermAttribute charTerm = tokens.getAttribute(CharTermAttribute.class);
				word.set(charTerm.toString() + "\t" + parentPath);
				context.write(word, new Text(name));//词和类做key， 文件名做value
			}
			analyzer.close();
		};
	}

	public static class MyPartitioner extends HashPartitioner<Text, IntWritable> {
		public int getPartition(Text key, IntWritable value, int numReduceTasks) {

			String term = new String();
			term = key.toString().split("\t")[0];
			return super.getPartition(new Text(term), value, numReduceTasks);
		}
	}

	public static class Combine extends Reducer<Text, Text, Text, Text> {
		protected void reduce(Text key, Iterable<Text> values, Context context)
				throws java.io.IOException, InterruptedException {

			String[] splits = key.toString().split("\t");
			if (splits.length != 2) {
				return;
			}
			Set<String> set = new HashSet<>();
			long count = 0;
			for (Text v : values) {
				if (!set.contains(v.toString())) {//同一个词在同一个文件出现只计数一次
					set.add(v.toString());
					count++;
				}
			}
			Text word = new Text(splits[0]);
			Text post = new Text(splits[1] + "#" + count);
			context.write(word, post);
		};
	}

	public static class Reduce extends Reducer<Text, Text, Text, Text> {

		protected void reduce(Text word, Iterable<Text> values, Context context)
				throws java.io.IOException, InterruptedException {

			long sum = 0; // sum of this word
			
			//合并词在同一个类的计数
			Map<String, Long> mapSet = new HashMap<>();
			for (Text v : values) {
				// docCount++;
				String temp = v.toString();
				String[] strings = temp.split("#");
				if (strings.length != 2) {
					return;
				}
				long count = Long.parseLong(strings[1]);
				sum += count;
				if(mapSet.containsKey(strings[0])){
					long old = mapSet.get(strings[0]);
					mapSet.remove(strings[0]);
					mapSet.put(strings[0], old+count);
				}
				else {
					mapSet.put(strings[0], count);
				}
			}
			//计算该词的信息增益
			double InfoGain = 0.0;//信息增益
			double Entropy = -20*0.05*log2(0.05); //信息熵
			double N = 20000.0-2264.0; //总文档数
			double ClassFileNum = 887.0;	//每个类别下文档数
			double nSum = N - (double)sum;
			double Pt = 0.0;
			double Pnt = ((double)(20-mapSet.size()))*(ClassFileNum/nSum)*log2(ClassFileNum/nSum);
			
			for (Entry<String, Long> entry: mapSet.entrySet()){
				Pt += ((double)entry.getValue())/((double)sum);
				Pnt += ((double)(ClassFileNum-entry.getValue()))/(nSum);
//		sub.append(entry.getKey()+":"+entry.getValue()).append(";");
			}
			
			mapSet.clear();
			InfoGain = Entropy + Pt*(((double)sum)/N) + Pnt*(nSum/N);
			
			Text postings = new Text(/*"InfoGain:" +*/ Double.toString(InfoGain) + "-" + Long.toString(sum));
			//InfoGain-Sum
			context.write(word, postings);
		};
	}

	public static double log2(double a) {
		return Math.log(a)/Math.log(2.0);
	}
	
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
		conf.set("cache", args[0].split("/")[1]);
		Job job1 = new Job(conf, "Task1");
		job1.setJarByClass(Index.class);
		// set custom map, reduce and combine.
		job1.setMapperClass(MyMap.class);
		job1.setPartitionerClass(MyPartitioner.class);
		job1.setCombinerClass(Combine.class);
		job1.setReducerClass(Reduce.class);

		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(Text.class);
		job1.addCacheFile(new Path(args[0]).toUri()); //加入全局文件
		
		
		// the parameter.
		FileInputFormat.addInputPath(job1, new Path(args[1]));
		FileInputFormat.setInputDirRecursive(job1, true);
		FileOutputFormat.setOutputPath(job1, new Path(args[2]));
		// wait and print the process, return true when finish successfully.
		//System.exit(job.waitForCompletion(true) ? 0 : 1);
		  job1.waitForCompletion(true);
	}
}
