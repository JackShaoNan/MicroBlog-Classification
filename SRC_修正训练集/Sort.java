package Final;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

//import Final.Index.Combine;
import Final.Index.MyMap;
//import Final.Index.MyPartitioner;
import Final.Index.Reduce;

public class Sort {
	public static class SortMap extends Mapper<LongWritable, Text, DoubleWritable, Text> {
		
		protected void map(LongWritable key, Text value, Context context)
				throws java.io.IOException, InterruptedException {
				DoubleWritable outkey = new DoubleWritable();
				Text outvalue = new Text();
				String input =value.toString();
				String split[] =input .split("\t");
				if(split[1].contains("-")){//InfoGain+sum
					String temp[] = split[1].split("-");
					//temp[0]为 split[0]代表词对应的InfoGain,temp[1]代表词对应的出现文件数sum
					outvalue.set(split[0]+"-"+temp[1]);
					outkey.set(Double.parseDouble(temp[0]));
					context.write(outkey, outvalue);
				}
		};
	}

	
	public static class SortReduce extends Reducer<DoubleWritable, Text, Text, Text> {

		private static int No = 0;
		protected void reduce(DoubleWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			Text outkey = new Text();
			Text outvalue = new Text();
			for(Text value:values){
				String temp = value.toString();
				String split[] = temp.split("-"); 
				//split[0]是特征词，split[1]是Sum,
				//key是InfoGain
				if(Double.parseDouble(key.toString())>0.5||Double.parseDouble(split[1].toString())>4.0){
					//取0.5为5W+全局特征词;		取1.0为2W+
					No++;
					outkey.set(split[0]);
					outvalue.set(/*key.toString()+","+*/No+","+split[1]);
					context.write(outkey, outvalue);
				}
			}
		};
	}

	public static double log2(double a) {
		return Math.log(a)/Math.log(2.0);
	}
	
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
//		conf.set("cache", args[2].split("/")[1]);
		Job job1_5 = new Job(conf, "Task1.5");
		job1_5.setJarByClass(Sort.class);
		// set custom map, reduce and combine.
		job1_5.setMapperClass(SortMap.class);
		//job.setPartitionerClass(MyPartitioner.class);
		//job.setCombinerClass(Combine.class);
		job1_5.setReducerClass(SortReduce.class);

		job1_5.setMapOutputKeyClass(DoubleWritable.class);
		job1_5.setMapOutputValueClass(Text.class);
		job1_5.setOutputKeyClass(Text.class);
		job1_5.setOutputValueClass(Text.class);
//		job.addCacheFile(new Path(args[2]).toUri());
		
		
		// the parameter.
		FileInputFormat.addInputPath(job1_5, new Path(args[0]));
		FileInputFormat.setInputDirRecursive(job1_5, true);
		FileOutputFormat.setOutputPath(job1_5, new Path(args[1]));
		// wait and print the process, return true when finish successfully.
		//System.exit(job1_5.waitForCompletion(true) ? 0 : 1);
		   job1_5.waitForCompletion(true);
	}
}
