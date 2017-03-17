/*
 * LensKit, an open source recommender systems toolkit.
 * Copyright 2010-2016 LensKit Contributors.  See CONTRIBUTORS.md.
 * Work on LensKit has been funded by the National Science Foundation under
 * grants IIS 05-34939, 08-08692, 08-12148, and 10-17697.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
package org.lenskit.slim;

import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.ratings.RatingVectorPDAO;
import org.lenskit.results.Results;
import org.lenskit.util.keys.Long2DoubleSortedArrayMap;
import org.lenskit.util.math.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SlimScorer extends AbstractItemScorer {
    private static final Logger logger = LoggerFactory.getLogger(SlimScorer.class);

    protected final SlimModel model;
    private final RatingVectorPDAO rvDAO;


    @Inject
    public SlimScorer(SlimModel m,
                      RatingVectorPDAO dao) {
        model = m;
        rvDAO = dao;
    }

    /**
     * Score items for a user.
     * @param user The user ID.
     * @param items The score vector.
     */
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        logger.debug("scoring {} items for user {} with details", items.size(), user);
        Long2DoubleMap ratings = Long2DoubleSortedArrayMap.create(rvDAO.userRatingVector(user));
        List<Result> results = new ArrayList<>();

        for (long item: items) {
            Long2DoubleMap weight = model.getWeights(item);
            double score = Vectors.dotProduct(ratings, weight);
            results.add(Results.create(item, score));
        }
        return Results.newResultMap(results);
    }

}
