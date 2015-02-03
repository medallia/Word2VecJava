package com.medallia.word2vec.util;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Simple class for storing two arbitrary objects in one.
 *
 * @param <K> the type of the first value
 * @param <V> the type of the second value
 */
public class Pair<K, V> implements Map.Entry<K, V>, Serializable {
	
	/** @see Serializable */
	private static final long serialVersionUID = 1L;

	/** The first item in the pair. */
	public final K first;
	/** The second item in the pair. */
	public final V second;
	
	/** Creates a new instance of Pair */
	protected Pair(K first, V second) {
		this.first = first;
		this.second = second;
	}

	/** Type-inferring constructor */
	public static <X,Y> Pair<X,Y> cons(X x, Y y) { return new Pair<X,Y>(x,y); }
	
	/** Type-inferring constructor for pairs of the same type, which can optionally be swapped */
	public static <X> Pair<X,X> cons(X x, X y, boolean swapped) { return swapped ? new Pair<X, X>(y, x) : new Pair<X, X>(x, y); }

	@Override
	public int hashCode() {
		// Compute by hand instead of using Encoding.combineHashes for improved performance
		return (first == null ? 0 : first.hashCode() * 13) + (second == null ? 0 : second.hashCode() * 17);
	}
	
	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		if (o == null || !getClass().equals(o.getClass()))
			return false;
		
		Pair<?,?> op = (Pair<?,?>) o;
		return Objects.equals(op.first, first) && Objects.equals(op.second, second);
	}

	/** @return {@link #first}; needed because String Templates have 'first' as a reserved word. */
	public K getOne() { return first; }
	/** @return {@link #first} */
	public K getFirst() { return first; }
	/** @return {@link #second} */
	public V getSecond() { return second; }
	
	@Override public String toString() {
		return "Pair<"+first+","+second+">";
	}
	
	/** @return a list with the two elements from this pair, regardless of whether they are null or not */
	public List<Object> asList() {
		return Lists.newArrayList(first, second);
	}
	
	/**
	 * @return a list of key/value pairs (keys are at even indices, values at odd) taken from the
	 *         given array, whose length must be even.
	 */
	@SafeVarargs
	public static <X> List<Pair<X, X>> fromPairs(X... args) {
		if (Common.isOdd(args.length))
			throw new IllegalArgumentException("Array length must be even: " + args.length);
		
		List<Pair<X, X>> l = new ArrayList<>(args.length / 2);
		for (int i = 0; i < args.length; i += 2)
			l.add(Pair.cons(args[i], args[i + 1]));
		return l;
	}
	
	/**
	 * Converts a Map to a List of pairs.
	 * Each entry in the map results in a Pair in the returned list.
	 */
	public static <X, Y> List<Pair<X, Y>> fromMap(Map<X, Y> m) {
		List<Pair<X, Y>> l = new ArrayList<>();
		for (Map.Entry<X, Y> me : m.entrySet()) {
			l.add(Pair.cons(me.getKey(), me.getValue()));
		}
		return l;
	}
	
	private static <X, Y, C extends Collection<Pair<X, Y>>> C fromMapFlatten(C c, Map<? extends X, ? extends Collection<? extends Y>> m) {
		for (Map.Entry<? extends X, ? extends Collection<? extends Y>> me : m.entrySet()) {
			for (Y y : me.getValue())
				c.add(Pair.<X, Y>cons(me.getKey(), y));
		}
		return c;
	}
	
	@Override public K getKey() { return first; }
	@Override public V getValue() { return second; }
	@Override public V setValue(V value) { throw new UnsupportedOperationException(); }
	
	/** Method that allows Pair to be used directly by the Setup system (wtf) */
	public String getName() { return String.valueOf(second); }

	/** @return a reversed version of this pair */
	public Pair<V, K> swapped() { return Pair.cons(second, first); }

	/** @return {@link Function} which performs a {@link #swapped()} */
	public static <K, V> Function<Pair<K, V>, Pair<V, K>> swappedFunction() {
		return new Function<Pair<K, V>, Pair<V, K>>() {
			@Override public Pair<V, K> apply(Pair<K, V> p) {
				return p.swapped();
			}
		};
	}
	
	/**
	 * @return {@link Ordering} which compares the first value of the pairs.
	 * Pairs with equal first value will be considered equivalent independent of the second value
	 */
	public static <X extends Comparable<? super X>> Ordering<Pair<X, ?>> firstComparator() {
		return new Ordering<Pair<X, ?>>() {
			@Override public int compare(Pair<X, ?> o1, Pair<X, ?> o2) {
				return Compare.compare(o1.first, o2.first);
			}
		};
	}
	
	/**
	 * @return {@link Ordering} which compares the second value of the pairs.
	 * Pairs with equal second value will be considered equivalent independent of the first value
	 */
	public static <Y extends Comparable<? super Y>> Ordering<Pair<?, Y>> secondComparator() {
		return new Ordering<Pair<?, Y>>() {
			@Override public int compare(Pair<?, Y> o1, Pair<?, Y> o2) {
				return Compare.compare(o1.second, o2.second);
			}
		};
	}
	
	/** @return {@link Ordering} which compares both values of the {@link Pair}s, with the first taking precedence. */
	public static <X extends Comparable<? super X>, Y extends Comparable<? super Y>> Ordering<Pair<X,Y>> firstThenSecondComparator() {
		return new Ordering<Pair<X,Y>>() {
			@Override public int compare(Pair<X, Y> o1, Pair<X, Y> o2) {
				int k = Compare.compare(o1.first, o2.first);
				if (k == 0) k = Compare.compare(o1.second, o2.second);
				return k;
			}
		};
	}
	
	/** @return {@link Ordering} which compares both values of the {@link Pair}s, with the second taking precedence. */
	public static <X extends Comparable<? super X>, Y extends Comparable<? super Y>> Ordering<Pair<X,Y>> secondThenFirstComparator() {
		return new Ordering<Pair<X,Y>>() {
			@Override public int compare(Pair<X, Y> o1, Pair<X, Y> o2) {
				int k = Compare.compare(o1.second, o2.second);
				if (k == 0)
					k = Compare.compare(o1.first, o2.first);
				return k;
			}
		};
	}
	
	/**
	 * Pair comparator that applies the given {@link Comparator} to the first value of the pairs
	 */
	public static <X, Y> Comparator<Pair<X,Y>> firstComparator(final Comparator<? super X> comp) {
		return new Comparator<Pair<X,Y>>() {
			@Override public int compare(Pair<X, Y> o1, Pair<X, Y> o2) {
				return comp.compare(o1.first, o2.first);
			}
		};
	}

	/**
	 * Pair comparator that applies the given {@link Comparator} to the second value of the pairs
	 */
	public static <X, Y> Comparator<Pair<X,Y>> secondComparator(final Comparator<? super Y> comp) {
		return new Comparator<Pair<X,Y>>() {
			@Override public int compare(Pair<X, Y> o1, Pair<X, Y> o2) {
				return comp.compare(o1.second, o2.second);
			}
		};
	}
	
	/**
	 * Pair comparator that compares both values of the pairs, with the first taking
	 * precedence; the order is reversed for the first value only.
	 */
	public static <X extends Comparable<? super X>, Y extends Comparable<? super Y>> Comparator<Pair<X,Y>> bothFirstReversedComparator() {
		return new Comparator<Pair<X,Y>>() {
			@Override public int compare(Pair<X, Y> o1, Pair<X, Y> o2) {
				int k = Compare.compare(o2.first, o1.first);
				if (k == 0) k = Compare.compare(o1.second, o2.second);
				return k;
			}
		};
	}
	
	private static <X, Y> Map<X, Y> fillMap(Map<X, Y> m, Iterable<? extends Pair<? extends X, ? extends Y>> pairs) {
		for (Pair<? extends X, ? extends Y> p : pairs) {
			m.put(p.first, p.second);
		}
		return m;
	}

	/** @return the combination of all the elements in each collection. For instance if the first collection is
	 * {@code [1, 2, 3]}, and the second one is {@code [a, b]}, then the result is {@code [(1, a), (1, b), (2, a), ...]}
	 */
	@SuppressWarnings("unchecked")
	public static <X, Y> List<Pair<X, Y>> cartesianProduct(Collection<X> c1, Collection<Y> c2) {
		return FluentIterable.from(Sets.cartesianProduct(ImmutableSet.copyOf(c1), ImmutableSet.copyOf(c2)))
				.transform(new Function<List<Object>, Pair<X, Y>>() {
					@Override public Pair<X, Y> apply(List<Object> objs) {
						X x = (X) objs.get(0);
						Y y = (Y) objs.get(1);
						return Pair.<X, Y>cons(x, y);
					}
				})
				.toList();
	}
	
	/** @return the elements at equal indices in the two lists, which must be of the same
	 * length, as pairs.
	 */
	public static <X, Y> List<Pair<X, Y>> zip(Collection<X> c1, Collection<Y> c2) {
		return zip(c1, c2, new ArrayList<Pair<X, Y>>(c1.size()), false);
	}
	
	/** @return the elements at equal indices in the two lists, which must be of the same
	 * length, as pairs.
	 */
	public static <X, Y> List<Pair<X, Y>> zip(X[] a1, Y[] a2) {
		return zip(ImmutableList.copyOf(a1), ImmutableList.copyOf(a2));
	}

	/**
	 * @return the elements at equal indices in the two lists, which must be of
	 *         the same length, as pairs, without duplicates removed from the
	 *         first list
	 */
	public static <X, Y> List<Pair<X, Y>> zipUnique(Collection<X> c1, Collection<Y> c2) {
		return zip(c1, c2, new ArrayList<Pair<X, Y>>(), true);
	}

	private static <X, Y> List<Pair<X, Y>> zip(Collection<X> c1, Collection<Y> c2, List<Pair<X, Y>> output, boolean uniqueKeys) {
		int size = c1.size();
		if (size != c2.size())
			throw new IllegalArgumentException("Collections must be of same size: " + size + ", " + c2.size());

		Set<X> set = uniqueKeys ? new HashSet<X>() : null;
		Iterator<X> it1 = c1.iterator();
		Iterator<Y> it2 = c2.iterator();

		while (it1.hasNext() && it2.hasNext()) {
			X x = it1.next();
			Y y = it2.next();
			if (set == null || set.add(x))
				output.add(Pair.cons(x, y));
		}
		return output;
	}

	/** 
	 * @return the elements at equal indices of the two list as pairs. The number of elements in the result list
	 * is the minimum of the given iterable
	 * of different size only elements at indices
	 *  present on the first {@link Iterable} are used.
	 */
	public static <X, Y> Iterable<Pair<X, Y>> zipInner(final Iterable<X> first, final Iterable<Y> second) {
		return new Iterable<Pair<X, Y>>() {
			@Override public Iterator<Pair<X, Y>> iterator() {
				final Iterator<X> x = first.iterator();
				final Iterator<Y> y = second.iterator();
				return new Iterator<Pair<X, Y>>() {
					@Override public boolean hasNext() {
						return x.hasNext() && y.hasNext();
					}

					@Override
					public Pair<X, Y> next() {
						return Pair.cons(x.next(), y.next());
					}

					@Override
					public void remove() {
						x.remove();
						y.remove();
					}
					
				};
			}
			
		};
	}

	/** @return {@link Function} which retrieves the second of the pair */
	public static <V> Function<Pair<?, V>, V> retrieveSecondFunction() {
		return new Function<Pair<?, V>, V>() {
			@Override
			public V apply(Pair<?, V> p) {
				return p.second;
			}
		};
	}

	/** @return {@link Iterable} of second element in pair */
	public static <K, V> Iterable<V> unzipSecond(Iterable<Pair<K, V>> pairs) {
		return Iterables.transform(pairs, Pair.<V>retrieveSecondFunction());
	}

	/** @return {@link Function} which maps the value of each pair through the given {@link Function} */
	public static <K, V, V2> Function<Pair<K, V>, Pair<K, V2>> mapValues(final Function<? super V, V2> func) {
		return new Function<Pair<K, V>, Pair<K, V2>>() {
			@Override public Pair<K, V2> apply(Pair<K, V> p) {
				return Pair.cons(p.first, func.apply(p.second));
			}
		};
	}

	/** @return the first value, or null if the pair is null */
	public static <K> K firstOrNull(Pair<K, ?> pair) {
		return pair != null ? pair.first : null;
	}
	
	/** @return the second value, or null if the pair is null */
	public static <V> V secondOrNull(Pair<?, V> pair) {
		return pair != null ? pair.second : null;
	}
	
	/** @return {@link Predicates} which filters only on the first value */
	public static <K, V> Predicate<Pair<K, V>> getFirstPredicate(final Predicate<? super K> pred) {
		return new Predicate<Pair<K, V>>() {
			@Override public boolean apply(Pair<K, V> pair) {
				return pred.apply(pair.first);
			}
		};
	}
	
	/** @return {@link Predicates} which filters only on the second value */
	public static <K, V> Predicate<Pair<K, V>> getSecondPredicate(final Predicate<? super V> pred) {
		return new Predicate<Pair<K, V>>() {
			@Override public boolean apply(Pair<K, V> pair) {
				return pred.apply(pair.second);
			}
		};
	}
	
	/** @return {@link Predicates} which accepts a pair only if both values are accepted */
	public static <K, V> Predicate<Pair<K, V>> getAndPredicate(final Predicate<? super K> firstPred, final Predicate<? super V> secondPred) {
		return new Predicate<Pair<K, V>>() {
			@Override public boolean apply(Pair<K, V> pair) {
				return firstPred.apply(pair.first) && secondPred.apply(pair.second);
			}
		};
	}
	
	/** @return {@link Predicates} which accepts a pair if either values is accepted */
	public static <K, V> Predicate<Pair<K, V>> getOrPredicate(final Predicate<? super K> firstPred, final Predicate<? super V> secondPred) {
		return new Predicate<Pair<K, V>>() {
			@Override public boolean apply(Pair<K, V> pair) {
				return firstPred.apply(pair.first) || secondPred.apply(pair.second);
			}
		};
	}

	/**
	 * @return {@link ImmutableList} containing all values paired with their applied value
	 * through the function
	 */
	public static <X, Y> ImmutableList<Pair<X, Y>> toPairList(Iterable<X> values, Function<X, Y> func) {
		ImmutableList.Builder<Pair<X, Y>> result = ImmutableList.builder();
		for (X x : values)
			result.add(Pair.cons(x, func.apply(x)));
		return result.build();
	}

}
