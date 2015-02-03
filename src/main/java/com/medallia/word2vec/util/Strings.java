package com.medallia.word2vec.util;

/**
 * Various utility functions for working with String objects
 */
public class Strings {
	/** @see {@link Strings#formatEnum(Enum)} */
	public static String formatEnum(Enum<?> enumValue) {
		return capitalizeFirstCharacterLowercaseRest(enumValue.name().replace('_', ' '));
	}

	private static String capitalizeFirstCharacterLowercaseRest(String s) {
		if (!hasContent(s)) return s;
		return s.substring(0, 1).toUpperCase() + s.substring(1).toLowerCase();
	}

	/** @return <code>true</code> if the string is not <code>null</code> and has non-zero trimmed length; <code>false</code> otherwise */
	public static boolean hasContent(String s) {
		return hasContent(s, true);
	}

	/**
	 * @param trim true if the string should be trimmed
	 * @return <code>true</code> if the string is not <code>null</code> and has non-zero trimmed length; <code>false</code> otherwise */
	public static boolean hasContent(String s, boolean trim) {
		return s != null && !(trim ? s.trim() : s).isEmpty();
	}

	/**
	 * Join the toString of each object element into a single string, with
	 * each element separated by the given sep (which can be empty).
	 */
	public static String joinObjects(String sep, Iterable<?> l) {
		return sepList(sep, l, -1);
	}

	/** Same as sepList with no wrapping */
	public static String sepList(String sep, Iterable<?> os, int max) {
		return sepList(sep, null, os, max);
	}

	/** @return The concatenation of toString of the objects obtained from the iterable, separated by sep, and if max
	 * is > 0 include no more than that number of objects. If wrap is non-null, prepend and append each object with it
	 */
	public static String sepList(String sep, String wrap, Iterable<?> os, int max) {
		StringBuilder sb = new StringBuilder();
		String s = "";
		if (max == 0) max = -1;
		for (Object o : os) {
			sb.append(s); s = sep;
			if (max-- == 0) { sb.append("..."); break; }
			if (wrap != null) sb.append(wrap);
			sb.append(o);
			if (wrap != null) sb.append(wrap);
		}
		return sb.toString();
	}
}
