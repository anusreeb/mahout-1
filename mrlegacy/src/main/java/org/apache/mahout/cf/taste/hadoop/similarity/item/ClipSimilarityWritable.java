 /**
	 * This has a clip_id and a similarity measure. This will be written as the value field
	 * for every clip_id. If the clip_id is the key, an object of the type ClipSimilarityWritable will have
	 * the id of a similar clip and the 
	 */

package org.apache.mahout.cf.taste.hadoop.similarity.item;

	
	import java.io.DataInput;
	import java.io.DataOutput;
	import java.io.IOException;

	import org.apache.hadoop.io.WritableComparable;
	import org.apache.mahout.math.Varint;


	public class ClipSimilarityWritable implements WritableComparable<ClipSimilarityWritable>, Cloneable{
		
		private long clipId;
		private double similarity;
		
		public ClipSimilarityWritable(){
			
		}
		
		public ClipSimilarityWritable(long clipId,double similarity){
			this.clipId=clipId;
			this.similarity=similarity;
		}
		
		public long getClipId(){
			return clipId;
		}
		
		@Override
		public void readFields(DataInput in) throws IOException {
			clipId=Varint.readSignedVarLong(in);
			similarity=in.readDouble();
		}
		
		@Override
		public void write(DataOutput out) throws IOException {
			Varint.writeSignedVarLong(clipId, out);
			out.writeDouble(similarity);
		}
		
		@Override
		public int compareTo(ClipSimilarityWritable that) {
			return compare(this.similarity,that.similarity);
		}
		
		private static int compare(double a,double b){
			return a < b ? -1 : a > b ? 1 : 0;
		}
		
		public ClipSimilarityWritable clone(){
			return new ClipSimilarityWritable(clipId,similarity);
		}
		
		public String toString(){
			return similarity+":"+clipId;
		}

	}


