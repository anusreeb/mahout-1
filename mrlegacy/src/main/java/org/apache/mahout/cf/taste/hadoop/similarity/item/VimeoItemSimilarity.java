/**
 * This is Mahout's ItemSimilarity implementation with Vimeo Specific Tweaks
 * 
 * Here's where the tweak list gets documented
 * 
 * 1.Let the reducer output clip_id and a list of ClipSimilarityWritable objects instead of 
 * EntityWritable and similarty 
 */
package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.preparation.PreparePreferenceMatrixJob;
import org.apache.mahout.cf.taste.hadoop.similarity.item.TopSimilarItemsQueue;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItem;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;
import org.apache.mahout.math.map.OpenIntLongHashMap;

public class VimeoItemSimilarity extends AbstractJob {
	
	public static final String ITEM_ID_INDEX_PATH_STR = VimeoItemSimilarity.class.getName() + ".itemIDIndexPathStr";
	  public static final String MAX_SIMILARITIES_PER_ITEM = VimeoItemSimilarity.class.getName() + ".maxSimilarItemsPerItem";

	  private static final int DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM = 100;
	  private static final int DEFAULT_MAX_PREFS = 500;
	  private static final int DEFAULT_MIN_PREFS_PER_USER = 1;

	  public static void main(String[] args) throws Exception {
	    ToolRunner.run(new VimeoItemSimilarity(), args);
	  }

	@Override
	public int run(String[] args) throws Exception {
		addInputOption();
	    addOutputOption();
	    addOption("similarityClassname", "s", "Name of distributed similarity measures class to instantiate, " 
	        + "alternatively use one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')');
	    addOption("maxSimilaritiesPerItem", "m", "try to cap the number of similar items per item to this number "
	        + "(default: " + DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM + ')',
	        String.valueOf(DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM));
	    addOption("maxPrefs", "mppu", "max number of preferences to consider per user or item, " 
	        + "users or items with more preferences will be sampled down (default: " + DEFAULT_MAX_PREFS + ')',
	        String.valueOf(DEFAULT_MAX_PREFS));
	    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this "
	        + "(default: " + DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MIN_PREFS_PER_USER));
	    addOption("booleanData", "b", "Treat input as without pref values", String.valueOf(Boolean.FALSE));
	    addOption("threshold", "tr", "discard item pairs with a similarity value below this", false);
	    addOption("randomSeed", null, "use this seed for sampling", false);

	    Map<String,List<String>> parsedArgs = parseArguments(args);
	    if (parsedArgs == null) {
	      return -1;
	    }

	    String similarityClassName = getOption("similarityClassname");
	    int maxSimilarItemsPerItem = Integer.parseInt(getOption("maxSimilaritiesPerItem"));
	    int maxPrefs = Integer.parseInt(getOption("maxPrefs"));
	    int minPrefsPerUser = Integer.parseInt(getOption("minPrefsPerUser"));
	    boolean booleanData = Boolean.valueOf(getOption("booleanData"));

	    double threshold = hasOption("threshold")
	        ? Double.parseDouble(getOption("threshold")) : RowSimilarityJob.NO_THRESHOLD;
	    long randomSeed = hasOption("randomSeed")
	        ? Long.parseLong(getOption("randomSeed")) : RowSimilarityJob.NO_FIXED_RANDOM_SEED;

	    Path similarityMatrixPath = getTempPath("similarityMatrix");
	    Path prepPath = getTempPath("prepareRatingMatrix");

	    AtomicInteger currentPhase = new AtomicInteger();

	    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
	      ToolRunner.run(getConf(), new PreparePreferenceMatrixJob(), new String[] {
	        "--input", getInputPath().toString(),
	        "--output", prepPath.toString(),
	        "--minPrefsPerUser", String.valueOf(minPrefsPerUser),
	        "--booleanData", String.valueOf(booleanData),
	        "--tempDir", getTempPath().toString(),
	      });
	    }

	    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
	      int numberOfUsers = HadoopUtil.readInt(new Path(prepPath, PreparePreferenceMatrixJob.NUM_USERS), getConf());

	      ToolRunner.run(getConf(), new RowSimilarityJob(), new String[] {
	        "--input", new Path(prepPath, PreparePreferenceMatrixJob.RATING_MATRIX).toString(),
	        "--output", similarityMatrixPath.toString(),
	        "--numberOfColumns", String.valueOf(numberOfUsers),
	        "--similarityClassname", similarityClassName,
	        "--maxObservationsPerRow", String.valueOf(maxPrefs),
	        "--maxObservationsPerColumn", String.valueOf(maxPrefs),
	        "--maxSimilaritiesPerRow", String.valueOf(maxSimilarItemsPerItem),
	        "--excludeSelfSimilarity", String.valueOf(Boolean.TRUE),
	        "--threshold", String.valueOf(threshold),
	        "--randomSeed", String.valueOf(randomSeed),
	        "--tempDir", getTempPath().toString(),
	      });
	    }

	    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
	      Job mostSimilarItems = prepareJob(similarityMatrixPath, getOutputPath(), SequenceFileInputFormat.class,
	          SimilarItemsMapper.class, Text.class, Text.class,
	          SimilarItemsReducer.class, Text.class, Text.class, TextOutputFormat.class);
	      Configuration mostSimilarItemsConf = mostSimilarItems.getConfiguration();
	      mostSimilarItemsConf.set(ITEM_ID_INDEX_PATH_STR,
	          new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
	      mostSimilarItemsConf.setInt(MAX_SIMILARITIES_PER_ITEM, maxSimilarItemsPerItem);
	      boolean succeeded = mostSimilarItems.waitForCompletion(true);
	      if (!succeeded) {
	        return -1;
	      }
	    }

	    return 0;
	}
	
	public static class SimilarItemsMapper
     extends Mapper<IntWritable,VectorWritable,Text,Text> {

   
   private int maxSimilarItemsPerItem;

   @Override
   protected void setup(Context ctx) {
     Configuration conf = ctx.getConfiguration();
     maxSimilarItemsPerItem = conf.getInt(MAX_SIMILARITIES_PER_ITEM, -1);
//     indexItemIDMap = TasteHadoopUtils.readIDIndexMap(conf.get(ITEM_ID_INDEX_PATH_STR), conf);

     Preconditions.checkArgument(maxSimilarItemsPerItem > 0, "maxSimilarItemsPerItem must be greater then 0!");
   }

   @Override
   protected void map(IntWritable itemIDIndexWritable, VectorWritable similarityVector, Context ctx)
     throws IOException, InterruptedException {

     int itemIDIndex = itemIDIndexWritable.get();

     TopSimilarItemsQueue topKMostSimilarItems = new TopSimilarItemsQueue(maxSimilarItemsPerItem);

     for (Vector.Element element : similarityVector.get().nonZeroes()) {
       SimilarItem top = topKMostSimilarItems.top();
       double candidateSimilarity = element.get();
       if (candidateSimilarity > top.getSimilarity()) {
//         top.set(indexItemIDMap.get(element.index()), candidateSimilarity);
    	   top.set(element.index(), candidateSimilarity);
         topKMostSimilarItems.updateTop();
       }
     }
     
//     long itemID = indexItemIDMap.get(itemIDIndex);
     for (SimilarItem similarItem : topKMostSimilarItems.getTopItems()) {
       long otherItemID = similarItem.getItemID();
       if (itemIDIndex < otherItemID) {
    	 Text key=new Text(Long.toString(itemIDIndex)+":"+Long.toString(otherItemID).toString());
    	 Text value=new Text(Double.toString(similarItem.getSimilarity()));
    	 
         ctx.write(key,value);
       } else {
    	  Text key=new Text(Long.toString(otherItemID)+":"+Long.toString(itemIDIndex).toString());
      	 Text value=new Text(Double.toString(similarItem.getSimilarity()));
         ctx.write(key,value);
       }
     }
//     long itemID = indexItemIDMap.get(itemIDIndex);
//     for (SimilarItem similarItem : topKMostSimilarItems.getTopItems()) {
//       long otherItemID = similarItem.getItemID();
//       if (itemID < otherItemID) {
//    	 Text key=new Text(Long.toString(itemID)+":"+Long.toString(otherItemID).toString());
//    	 Text value=new Text(Double.toString(similarItem.getSimilarity()));
//    	 
//         ctx.write(key,value);
//       } else {
//    	  Text key=new Text(Long.toString(otherItemID)+":"+Long.toString(itemID).toString());
//      	 Text value=new Text(Double.toString(similarItem.getSimilarity()));
//         ctx.write(key,value);
//       }
//     }
   }
 }

 public static class SimilarItemsReducer
     extends Reducer<Text,Text,Text,Text> {
	 
	 private OpenIntLongHashMap indexItemIDMap;
	 
	 @Override
	   protected void setup(Context ctx) {
	     Configuration conf = ctx.getConfiguration();
	     indexItemIDMap = TasteHadoopUtils.readIDIndexMap(conf.get(ITEM_ID_INDEX_PATH_STR), conf);
	   }
	 
   @Override
   protected void reduce(Text key,Iterable<Text> values,Context ctx) throws IOException, InterruptedException
   {
	   long clip1=indexItemIDMap.get(Integer.parseInt(key.toString().split(":")[0]));
	   long clip2=indexItemIDMap.get(Integer.parseInt(key.toString().split(":")[1]));
	   Text newKey=new Text(Long.toString(clip1)+":"+Long.toString(clip2));
	   ctx.write(newKey, values.iterator().next());
   }
//   protected void reduce(Text key, Iterable<Text> values, Context ctx)
//     throws IOException, InterruptedException {
//	   
//     ctx.write(key,values.iterator().next() );
//   }
 }

}
