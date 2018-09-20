package Final;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

public class ClassFeatureVector {
	public static class CFVMap extends Mapper<LongWritable, Text, Text, Text> {
		public Map<String, String> wordToNo = new HashMap<>();

		// 为了方便后面计算贝叶斯，将特征词的编号换成其本身，因此设置一个词与编号的全局对应表
		protected void setup(Context context) throws IOException {
			Configuration configuration = context.getConfiguration();

			Path path = new Path(configuration.get("wordToNo"));
			BufferedReader bufferedReader = new BufferedReader(new FileReader(path.toString()));
			String line;
			while ((line = bufferedReader.readLine()) != null) {
				String[] temp = line.split("\t");
				wordToNo.put(temp[1].split(",")[0], temp[0]);// 词 编号
			}
			bufferedReader.close();
		}

		// 将输入从类与文件名作为key，特征向量为值，转变为类和词做key
		//输入格式： 类名		文件名		特征向量(特征词#在该文件中TF-IDF值)
		//输出格式：Outkey:类名  特征词     ;Outvalue:TF-IDF值   
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString();
			String[] strings = line.split("\t");// 类 文件名 词:值(多个)
			String[] post = strings[2].split(" ");
			Text Outvalue = new Text();
			for (String string : post) { 
				String[] temp = string.split("#");
				Text classAndWord = new Text(strings[0] + "\t" + wordToNo.get(temp[0]));
				context.write(classAndWord, new Text(temp[1]));//temp[0]为特征词编号,temp[1]为其对应TF-IDF值
			}
		}
	}

	// 按照类排序
	public static class CFVPartitioner extends HashPartitioner<Text, IntWritable> {
		public int getPartition(Text key, IntWritable value, int numReduceTasks) {

			String term = new String();
			term = key.toString().split("\t")[0];
			return super.getPartition(new Text(term), value, numReduceTasks);
		}
	}

	public static class CFVCombine extends Reducer<Text, Text, Text, Text> {
		protected void reduce(Text key, Iterable<Text> values, Context context)
				throws java.io.IOException, InterruptedException {
			String[] splits = key.toString().split("\t");
			if (splits.length != 2) {
				return;
			}

			double count = 0.0;
			for (Text v : values) {
				count = count + Double.parseDouble(v.toString()); //将其在该类的TF-IDF值相加
			}
			Text word = new Text(splits[0]);//类名
			Text post = new Text(splits[1] + "#" + Double.toString(count));//特征词#在该类中TF-IDF值之和
			context.write(word, post);
		};
	}

	// 最终转成类 特征词及其计数
	//Outkey为类名   Outvalue为特征词#该类下TF-IDF值
	public static class CFVReduce extends Reducer<Text, Text, Text, Text> {
		protected void reduce(Text word, Iterable<Text> values, Context context)
				throws java.io.IOException, InterruptedException {
			StringBuilder stringBuilder = new StringBuilder();
			for (Text text : values) {
				stringBuilder.append(text.toString() + " ");
			}
			context.write(word, new Text(stringBuilder.toString()));
		};//组织成类特征向量
	}

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();

		String[] wTN = args[0].split("/");
		conf.set("wordToNo", wTN[wTN.length - 1]);

		Job job3 = new Job(conf, "Task3");
		job3.setJarByClass(ClassFeatureVector.class);

		job3.setMapperClass(CFVMap.class);
		job3.setPartitionerClass(CFVPartitioner.class);
		job3.setCombinerClass(CFVCombine.class);
		job3.setReducerClass(CFVReduce.class);

		job3.setMapOutputKeyClass(Text.class);
		job3.setMapOutputValueClass(Text.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);

		job3.addCacheFile(new Path(args[0]).toUri());

		// the parameter.
		FileInputFormat.addInputPath(job3, new Path(args[1]));
		FileOutputFormat.setOutputPath(job3, new Path(args[2]));
		// wait and print the process, return true when finish successfully.
		//System.exit(job.waitForCompletion(true) ? 0 : 1);
		  job3.waitForCompletion(true);
	}
}
