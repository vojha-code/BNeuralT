package debaug;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.IntStream;

public class General {

	public static void main(String args[]) {
		int[] e = IntStream.range(0, 20).toArray();
		ArrayList<Integer> r = new ArrayList<Integer>();
		for(int i = 0; i< e.length;i++) {
			r.add(i);
		}

		int b = 3;
		int bc = e.length;
		int bs = -1;
		int count = 0;
		Collections.shuffle(r);// Shuffle //  System.out.println("Shuffled:"+randIndex);
		System.out.println(Arrays.toString(e));
		System.out.println(r);
		while(bc > 0) {
			System.out.print(bc +" : ");
			if(bc < b) {
				bs = bc;
			}else {
				bs = b;
			}
			System.out.print(bs +" : ");
			int[] ba = new int[bs];
			for(int i =0; i < bs;i++) {
				ba[i] = e[r.get(count + i)];
				System.out.print(" "+ba[i]);
			}
			System.out.println();
			bc = bc - b;
			count = count + bs;
		}
		System.out.println("="+ count);

		System.out.printf("GD: %s best @start: %.4f  @end: %.4f  cahnge: %d %s \n", "dfg", 2.2222, 3.255555, 5, java.time.LocalDateTime.now().toString());
	}
}
