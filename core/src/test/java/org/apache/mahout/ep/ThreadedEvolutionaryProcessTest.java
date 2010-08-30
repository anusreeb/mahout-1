package org.apache.mahout.ep;

import org.junit.Assert;
import org.junit.Test;

import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutionException;

public class ThreadedEvolutionaryProcessTest {
  @Test
  public void testOptimize() throws ExecutionException, InterruptedException {
    ThreadedEvolutionaryProcess ep = new ThreadedEvolutionaryProcess(50);
    State x = ep.optimize(new ThreadedEvolutionaryProcess.Function() {
      /**
       * Implements a skinny quadratic bowl.
       */
      @Override
      public double apply(double[] params) {
        Random rand = new Random(1);
        double sum = 0;
        int i = 0;
        for (double x : params) {
          x = (i + 1) * (x - i);
          i++;
          sum += x * x;
        }
        try {
          // variable delays to emulate a tricky function
          Thread.sleep((long) Math.floor(-2 * Math.log(1 - rand.nextDouble())));
        } catch (InterruptedException e) {
          // ignore interruptions
        }

        return -sum;
      }
    }, 5, 200, 2);

    System.out.printf("%s\n", ep);
    double[] r = x.getMappedParams();
    int i = 0;
    for (double v : r) {
      Assert.assertEquals(String.format(Locale.ENGLISH, "Coordinate %d", i), i, v, 0.02);
      i++;
    }
  }
}