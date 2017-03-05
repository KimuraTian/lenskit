package org.lenskit.slim;

/**
 * Created by tmc on 2/19/17.
 */

import com.google.common.collect.Maps;
import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongOpenHashBigSet;
import org.junit.Before;
import org.junit.Test;
import org.lenskit.util.math.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Map;
import java.util.Random;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;

public class NaiveCoordDestLinearRegrTest {
    private Long2DoubleMap y;
    private Map<Long, Long2DoubleMap> data;
    private Long2DoubleMap weights;
    final static Logger logger = LoggerFactory.getLogger(org.lenskit.slim.NaiveCoordDestLinearRegrTest.class);

    // create weights
    static Long2DoubleMap createWeights(int itemIdBound, int maxWeight) {
        Long2DoubleMap weights = new Long2DoubleOpenHashMap();
        Random r = new Random();
        for (long w = 0; w < itemIdBound; w++) {
            int maxW = r.nextInt(maxWeight);
            weights.put(w, maxW*r.nextDouble());
        }
        //logger.info("original weights is {} and size is {}", weights, weights.size());
        return weights;
    }
    // simulate user-item ratings matrix
    static Map<Long, Long2DoubleMap> createModel(int itemIdBound, Long2DoubleMap weights) {
        Random r = new Random();
        Map<Long, Long2DoubleMap> simulatedData = Maps.newHashMap();
        Long2DoubleMap labels = new Long2DoubleOpenHashMap();

        int userNum = 2000;
        int maxRatingNum = 50; // each user's max rating number (less than itemIdBound)
        int userIdBound = 5000; // greater than userNum (userId not necessarily ranging from 0 to userNum)
        double ratingRange = 5.0;

        int size = 0;
        while (size < userNum) {
            long userId = r.nextInt(userIdBound);
            int userRatingNum = r.nextInt(maxRatingNum);
            Long2DoubleMap userRatings = new Long2DoubleOpenHashMap();
            int i = 0;
            while (i < userRatingNum) {
                long itemId = r.nextInt(itemIdBound);
                double rating = ratingRange * r.nextDouble();
                userRatings.put(itemId, rating);
                i = userRatings.keySet().size();
            }
            double label = Vectors.dotProduct(userRatings, weights) + r.nextGaussian();
            labels.put(userId, label);
            simulatedData.put(userId, userRatings);
            size = simulatedData.keySet().size();
        }
        simulatedData.put((long)userIdBound, labels);
        //logger.info("rating matrix is {}", simulatedData);
        return simulatedData;
    }

    @Before
    public void buildModel() {
        int itemNum = 1000;
        int maxWeight = 10;
        this.weights = createWeights(itemNum, maxWeight);
        Map<Long, Long2DoubleMap> temp = Maps.newHashMap(createModel(itemNum, weights));
        LongOpenHashBigSet userIdSet = new LongOpenHashBigSet(temp.keySet());
        long maxUserId = Collections.max(userIdSet);
        this.y = new Long2DoubleOpenHashMap(temp.get(maxUserId));
        temp.remove(maxUserId);
        this.data = Maps.newHashMap(temp);
        Map<Long, Long2DoubleMap> dataT = LinearRegressionHelper.transposeMap(data);
        LongOpenHashBigSet itemKeySet = new LongOpenHashBigSet(dataT.keySet());
        logger.info("item matrix size {} \n max item id is {} \n min item id is {} ",itemKeySet.size64(), Collections.max(itemKeySet), Collections.min(itemKeySet));
    }

    @Test
    public void testLabels() {
        int sizeOfy = y.size();
        assertThat(sizeOfy, equalTo(2000));
    }

    @Test
    public void testWeights() {
        int sizeOfWeights = weights.size();
        assertThat(sizeOfWeights, equalTo(1000));
    }

    @Test
    public void testData() {
        int sizeOfData = data.keySet().size();
        assertThat(sizeOfData, equalTo(2000));
    }

    @Test
    public void testLinearRegression() {

        // Naive update
        final long startTimeNaive = System.currentTimeMillis();
        NaiveCoordDestLinearRegression model = new NaiveCoordDestLinearRegression(2.0, 0.2, false, 100);
        Long2DoubleMap predictedW = model.fit(y, data);
        //Long2DoubleMap predictions = model.predict(data, predictedW);
        Long2DoubleMap residuals = model.computeResiduals(y, data, predictedW);
        //double lossFun = model.computeLossFunction(residuals, predictedW);
        //System.out.println(lossFun);
        final long endTimeNaive = System.currentTimeMillis();


        // Covariance update
        final long startTimeCov = System.currentTimeMillis();
        CovarianceUpdateCoordDestLinearRegression model_Cov = new CovarianceUpdateCoordDestLinearRegression(2.0, 0.2, false, 100);
        Long2DoubleMap predictedW_Cov = model_Cov.fit(y, data);
        //Long2DoubleMap predictions = model.predict(data, predictedW);
        Long2DoubleMap residuals_Cov = model_Cov.computeResiduals(y, data, predictedW_Cov);
        //double lossFun_Cov = model_Cov.computeLossFunction(residuals_Cov, predictedW_Cov);
        //System.out.println(lossFun);
        final long endTimeCov = System.currentTimeMillis();
        logger.info("running time naive {}", (endTimeNaive - startTimeNaive));
        logger.info("running time Cov {}", (endTimeCov - startTimeCov));


        logger.info("Naive predicted weights is {}", predictedW);
        Long2DoubleMap weightDiff = LinearRegressionHelper.addVectors(weights, Vectors.multiplyScalar(predictedW, -1.0));
        double weightDiffNorm = Vectors.euclideanNorm(weightDiff);
        logger.info("weight difference is {} \n and the norm is {}", weightDiff, weightDiffNorm);

        logger.info("Cov predicted weights is {} \n original weight is {}", predictedW_Cov, weights);
        Long2DoubleMap weightDiff_Cov = LinearRegressionHelper.addVectors(weights, Vectors.multiplyScalar(predictedW_Cov, -1));
        double weightDiffNorm_Cov = Vectors.euclideanNorm(weightDiff_Cov);
        logger.info("Cov update : weight difference is {} \n and the norm is {}", weightDiff_Cov, weightDiffNorm_Cov);
        Long2DoubleMap resOrig = model_Cov.computeResiduals(y, data, weights);
        double lossFunOrig = model_Cov.computeLossFunction(resOrig, weights);
        logger.info("Original loss function is {}", lossFunOrig);
        Long2DoubleMap nonzeroWeigt = LinearRegressionHelper.filterValues(predictedW_Cov, 0.0);
        logger.info("non-zero weight size {}", nonzeroWeigt.size());

    }

}
