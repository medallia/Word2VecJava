package com.medallia.word2vec.util;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.io.Serializable;
import java.io.StringWriter;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Simple utilities that in no way deserve their own class.
 */
public class Common {
	/**
	 * @param distance use 1 for our caller, 2 for their caller, etc...
	 * @return the stack trace element from where the calling method was invoked
	 */
	public static StackTraceElement myCaller(int distance) {
		// 0 here, 1 our caller, 2 their caller
		int index = distance + 1;
		try {
			StackTraceElement st[] = new Throwable().getStackTrace();
			// hack: skip synthetic caster methods
			if (st[index].getLineNumber() == 1) return st[index + 1];
			return st[index];
		} catch (Throwable t) {
			return new StackTraceElement("[unknown]","-","-",0);
		}
	}

	/** Serialize the given object into the given stream */
	public static void serialize(Serializable obj, ByteArrayOutputStream bout) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(bout);
			out.writeObject(obj);
			out.close();
		} catch (IOException e) {
			throw new IllegalStateException("Could not serialize " + obj, e);
		}
	}

	/**
	 * Read the file line for line and return the result in a list
	 * @throws IOException upon failure in reading, note that we wrap the underlying IOException with the file name
	 */
	public static List<String> readToList(File f) throws IOException {
		try (final Reader reader = asReaderUTF8Lenient(new FileInputStream(f))) {
			return readToList(reader);
		} catch (IOException ioe) {
			throw new IllegalStateException(String.format("Failed to read %s: %s", f.getAbsolutePath(), ioe), ioe);
		}
	}
	/** Read the Reader line for line and return the result in a list */
	public static List<String> readToList(Reader r) throws IOException {
		try ( BufferedReader in = new BufferedReader(r) ) {
			List<String> l = new ArrayList<>();
			String line = null;
			while ((line = in.readLine()) != null)
				l.add(line);
			return Collections.unmodifiableList(l);
		}
	}

	/** Wrap the InputStream in a Reader that reads UTF-8. Invalid content will be replaced by unicode replacement glyph. */
	public static Reader asReaderUTF8Lenient(InputStream in) {
		return new UnicodeReader(in, "utf-8");
	}

	/** Read the contents of the given file into a string */
	public static String readFileToString(File f) throws IOException {
		StringWriter sw = new StringWriter();
		IO.copyAndCloseBoth(Common.asReaderUTF8Lenient(new FileInputStream(f)), sw);
		return sw.toString();
	}

	/** @return true if i is an even number */
	public static boolean isEven(int i) { return (i&1)==0; }
	/** @return true if i is an odd number */
	public static boolean isOdd(int i) { return !isEven(i); }

	/** Read the lines (as UTF8) of the resource file fn from the package of the given class into a (unmodifiable) list of strings
	 * @throws IOException */
	public static List<String> readResource(Class<?> clazz, String fn) throws IOException {
		try (final Reader reader = asReaderUTF8Lenient(getResourceAsStream(clazz, fn))) {
			return readToList(reader);
		}
	}

	/** Get an input stream to read the raw contents of the given resource, remember to close it :) */
	public static InputStream getResourceAsStream(Class<?> clazz, String fn) throws IOException {
		InputStream stream = clazz.getResourceAsStream(fn);
		if (stream == null) {
			throw new IOException("resource \"" + fn + "\" relative to " + clazz + " not found.");
		}
		return unpackStream(stream, fn);
	}
	
	/** Get a file to read the raw contents of the given resource :) */
  public static File getResourceAsFile(Class<?> clazz, String fn) throws IOException {
    URL url = clazz.getResource(fn);
    if (url == null || url.getFile() == null) {
      throw new IOException("resource \"" + fn + "\" relative to " + clazz + " not found.");
    }
    return new File(url.getFile());
  }

	/**
	 * @throws IOException if {@code is} is null or if an {@link IOException} is thrown when reading from {@code is}
	 */
	public static InputStream unpackStream(InputStream is, String fn) throws IOException {
		if (is == null)
			throw new FileNotFoundException("InputStream is null for " + fn);

		switch (FilenameUtils.getExtension(fn).toLowerCase()) {
			case "gz":
				return new GZIPInputStream(is);
			default:
				return is;
		}
	}

	/** Read the lines (as UTF8) of the resource file fn from the package of the given class into a string */
	public static String readResourceToStringChecked(Class<?> clazz, String fn) throws IOException {
		try (InputStream stream = getResourceAsStream(clazz, fn)) {
			return IOUtils.toString(asReaderUTF8Lenient(stream));
		}
	}
}
