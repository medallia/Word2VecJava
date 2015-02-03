package com.medallia.word2vec.util;

/** Utility class for general comparison and equality operations. */
public class Compare {
	/**
	 * {@link NullPointerException} safe compare method; nulls are less than non-nulls.
	 */
	public static <X extends Comparable<? super X>> int compare(X x1, X x2) {
		if (x1 == null) return x2 == null ? 0 : -1;
		return x2 == null ? 1 : x1.compareTo(x2);
	}
}
