/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.decompositions

import scala.reflect.ClassTag
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import RLikeDrmOps._
import org.apache.log4j.Logger

object DQR {

  private val log = Logger.getLogger(DQR.getClass)

  /**
   * Distributed _thin_ QR. A'A must fit in a memory, i.e. if A is m x n, then n should be pretty
   * controlled (<5000 or so). <P>
   *
   * It is recommended to checkpoint A since it does two passes over it. <P>
   *
   * It also guarantees that Q is partitioned exactly the same way (and in same key-order) as A, so
   * their RDD should be able to zip successfully.
   */
  def dqrThin[K: ClassTag](A: DrmLike[K], checkRankDeficiency: Boolean = true): (DrmLike[K], Matrix) = {

    if (A.ncol > 5000)
      log.warn("A is too fat. A'A must fit in memory and easily broadcasted.")

    implicit val ctx = A.context

    val AtA = (A.t %*% A).checkpoint()
    val inCoreAtA = AtA.collect

    if (log.isDebugEnabled) log.debug("A'A=\n%s\n".format(inCoreAtA))

    val ch = chol(inCoreAtA)
    val inCoreR = (ch.getL cloned) t

    if (log.isDebugEnabled) log.debug("R=\n%s\n".format(inCoreR))

    if (checkRankDeficiency && !ch.isPositiveDefinite)
      throw new IllegalArgumentException("R is rank-deficient.")

    val bcastAtA = drmBroadcast(inCoreAtA)

    // Unfortunately, I don't think Cholesky decomposition is serializable to backend. So we re-
    // decompose A'A in the backend again.

    // Compute Q = A*inv(L') -- we can do it blockwise.
    val Q = A.mapBlock() {
      case (keys, block) => keys -> chol(bcastAtA).solveRight(block)
    }

    Q -> inCoreR
  }

}
