package Final;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class Precision {
	public static class PMap extends Mapper<LongWritable, Text, Text, Text> {
		
		protected void map(LongWritable key, Text value, Context context)
				throws java.io.IOException, InterruptedException {
			String line = value.toString();
			String[] strings = line.split("\t");
			String ActualClass = strings[0].split("@")[0];
			if(ActualClass.equals(strings[1])){//预测结果与真实结果一致
				context.write(new Text("precision"), new Text("1")); //预测正确，计1
			}
			else {
				context.write(new Text("precision"), new Text("0")); //预测错误，计0
			}
		}
	}
	
	 public static class PReduce extends Reducer<Text, Text, Text, Text>{
         
	        protected void reduce(Text word, Iterable<Text> values, Context context)
	                throws java.io.IOException ,InterruptedException {
	        	long sum = 0;
	        	long right = 0;
	        	for (Text text : values) {
					sum++;
					if(text.toString().equals("1")){
						right++;
					}
				}
	        	double precision = ((double)right)/((double)sum); //计算正确率
	        	context.write(word, new Text("Sum: "+Long.toString(sum)+";Right: "+Long.toString(right)+";Precison: "+Double.toString(precision)));
	        }
	 }
	 
	 public static void main(String[] args) throws Exception{
			
			Configuration conf = new Configuration();
			
	        Job job4 = new Job(conf, "Task4");
	        job4.setJarByClass(ClassFeatureVector.class);
	        
	        job4.setMapperClass(PMap.class);
	        job4.setReducerClass(PReduce.class);
	        
	        job4.setMapOutputKeyClass(Text.class);
	        job4.setMapOutputValueClass(Text.class);
	        job4.setOutputKeyClass(Text.class);
	        job4.setOutputValueClass(Text.class);
	        
	         //the parameter.
	        FileInputFormat.addInputPath(job4, new Path(args[0]));
	        FileOutputFormat.setOutputPath(job4, new Path(args[1]));
	         //wait and print the process, return true when finish successfully.
	        //System.exit(job.waitForCompletion(true) ? 0 : 1);
	        job4.waitForCompletion(true);
	   }
	 
}
