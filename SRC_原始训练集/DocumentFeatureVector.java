package Final;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class DocumentFeatureVector {
	public static class DFVMap extends Mapper<LongWritable, Text, Text, Text> {
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// string格式 word\t类@文件：TF-IDF值;
			String line = value.toString();
			String[] strings = line.split("\t");
			String[] post = strings[1].split(";");
			for (String string : post) {
				String[] temp = string.split("#");
				Text classAndDoc = new Text(temp[0]);
				Text wordAndVal = new Text(strings[0] + "#" + temp[1]);
				context.write(classAndDoc, wordAndVal);
			}
		}
	}

	public static class DFVReduce extends Reducer<Text, Text, Text, Text> {
		public int vectorNum = 200;// 每个文件保留的特征词数 $ 参数矫正处--2.5

		protected void reduce(Text word, Iterable<Text> values, Context context)
				throws java.io.IOException, InterruptedException {
			// 直接排序会崩溃，所以以时间换空间，只留11个，从大到小排列，每次只需比较最后一个，如果发生替换，需要重新排序
			ArrayList<String> tempVal = new ArrayList<>();
			for (Text v : values) {
				if (tempVal.size() > vectorNum) {
					double d1 = Double.parseDouble(tempVal.get(vectorNum).split("#")[1]);
					double d2 = Double.parseDouble(v.toString().split("#")[1]);
					if (d2 > d1) {
						tempVal.remove(vectorNum);
						tempVal.add(v.toString());
						Collections.sort(tempVal, new Comparator<String>() {

							@Override
							public int compare(String o1, String o2) {
								double d1 = Double.parseDouble(o1.split("#")[1]);
								double d2 = Double.parseDouble(o2.split("#")[1]);
								return -Double.compare(d1, d2);// 从大到小排
							}

						});
					}
				} else if (tempVal.size() < vectorNum) {
					tempVal.add(v.toString());
				} else {
					tempVal.add(v.toString());
					Collections.sort(tempVal, new Comparator<String>() {

						@Override
						public int compare(String o1, String o2) {
							double d1 = Double.parseDouble(o1.split("#")[1]);
							double d2 = Double.parseDouble(o2.split("#")[1]);
							return -Double.compare(d1, d2);// 从大到小排
						}

					});
				}
			}

			String key = word.toString();
			String[] strings = key.split("@");
			Text cAndD = new Text(strings[0] + "\t" + strings[1]);
			StringBuilder stringBuilder = new StringBuilder(tempVal.get(0));
			for (int i = 1; i < vectorNum && i < tempVal.size(); i++) {
				stringBuilder.append(" " + tempVal.get(i));
			}
			context.write(cAndD, new Text(stringBuilder.toString()));
			tempVal.clear();
		}
	}

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();

		Job job2_5 = new Job(conf, "Task2.5");
		job2_5.setJarByClass(DocumentFeatureVector.class);

		job2_5.setMapperClass(DFVMap.class);
		job2_5.setReducerClass(DFVReduce.class);

		job2_5.setMapOutputKeyClass(Text.class);
		job2_5.setMapOutputValueClass(Text.class);
		job2_5.setOutputKeyClass(Text.class);
		job2_5.setOutputValueClass(Text.class);
		// the parameter.
		FileInputFormat.addInputPath(job2_5, new Path(args[0]));
		FileOutputFormat.setOutputPath(job2_5, new Path(args[1]));
		// wait and print the process, return true when finish successfully.
		//System.exit(job2_5.waitForCompletion(true) ? 0 : 1);
		  job2_5.waitForCompletion(true);
	}

}
