package com.medallia.word2vec.util;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import org.apache.commons.io.IOUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
import java.io.OutputStream;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Field;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Static utility functions related to IO.
 */
public final class IO {
	
	/** a Comparator that orders {@link File} objects as by the last modified date */
	public static final Comparator<File> FILE_LAST_MODIFIED_COMPARATOR = new Comparator<File>() {
		@Override public int compare(File o1, File o2) {
			return Long.compare(o1.lastModified(), o2.lastModified());
		}};
	
	private static final int DEFAULT_BUFFER_SIZE = 1024 * 8; // 8K. Same as BufferedOutputStream, so writes are written-through with no extra copying
	
	private IO() { }

	/** @return the oldest file in the given collection using the last modified date or return null if the collection is empty */
	public static File getOldestFileOrNull(Collection<File> files) {
		if (files.isEmpty())
			return null;
		
		return Collections.min(files, FILE_LAST_MODIFIED_COMPARATOR);
	}

	/** Copy input to output and close the output stream before returning */
	public static long copyAndCloseOutput(InputStream input, OutputStream output) throws IOException {
		try (OutputStream outputStream = output) {
			return copy(input, outputStream);
		}
	}
	
	/** Copy input to output and close both the input and output streams before returning */
	public static long copyAndCloseBoth(InputStream input, OutputStream output) throws IOException {
		try (InputStream inputStream = input) {
			return copyAndCloseOutput(inputStream, output);
		}
	}

	/** Similar to {@link IOUtils#toByteArray(InputStream)} but closes the stream. */
	public static byte[] toByteArray(InputStream is) throws IOException {
		ByteArrayOutputStream bao = new ByteArrayOutputStream();
		IO.copyAndCloseBoth(is, bao);
		return bao.toByteArray();
	}
	
	/** Copy input to output; neither stream is closed */
	public static long copy(InputStream input, OutputStream output) throws IOException {
		byte[] buffer = new byte[DEFAULT_BUFFER_SIZE];
		long count = 0;
		int n = 0;
		while (-1 != (n = input.read(buffer))) {
			output.write(buffer, 0, n);
			count += n;
		}
		return count;
	}
	
	/** Copy input to output; neither stream is closed */
	public static int copy(Reader input, Writer output) throws IOException {
		char[] buffer = new char[DEFAULT_BUFFER_SIZE];
		int count = 0;
		int n = 0;
		while (-1 != (n = input.read(buffer))) {
			output.write(buffer, 0, n);
			count += n;
		}
		return count;
	}
	
	/** Copy input to output and close the output stream before returning */
	public static int copyAndCloseOutput(Reader input, Writer output) throws IOException {
		try {
			return copy(input, output);
		} finally {
			output.close();
		}
		
	}
	/** Copy input to output and close both the input and output streams before returning */
	public static int copyAndCloseBoth(Reader input, Writer output) throws IOException {
		try {
			return copyAndCloseOutput(input, output);
		} finally {
			input.close();
		}
	}
	
	/**
	 * Copy the data from the given {@link InputStream} to a temporary file and call the given
	 * {@link Function} with it; after the function returns the file is deleted.
	 */
	public static <X> X runWithFile(InputStream stream, Function<File, X> function) throws IOException {
		File f = File.createTempFile("run-with-file", null);
		try {
			try (FileOutputStream out = new FileOutputStream(f)) {
				IOUtils.copy(stream, out);
			}
			return function.apply(f);
		} finally {
			f.delete();
		}
	}

	/** @return the compressed (gzip) version of the given bytes */
	public static byte[] gzip(byte[] in) {
		try {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			GZIPOutputStream gz = new GZIPOutputStream(bos);
			gz.write(in);
			gz.close();
			return bos.toByteArray();
		} catch (IOException e) {
			throw new RuntimeException("Failed to compress bytes", e);
		}
	}

	/** @return the decompressed version of the given (compressed with gzip) bytes */
	public static byte[] gunzip(byte[] in) {
		try {
			GZIPInputStream gis = new GZIPInputStream(new ByteArrayInputStream(in));
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			IO.copyAndCloseOutput(gis, bos);
			return bos.toByteArray();
		} catch (IOException e) {
			throw new RuntimeException("Failed to decompress data", e);
		}
	}

	/** @return the compressed (gzip) version of the given object */
	public static byte[] gzipObject(Object object) {
		try (
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			GZIPOutputStream zos = new GZIPOutputStream(bos);
			ObjectOutputStream oos = new ObjectOutputStream(zos))
		{
			oos.writeObject(object);
			zos.close(); // Terminate gzip
			return bos.toByteArray();
		} catch (IOException e) {
			throw new RuntimeException("Failed to compress bytes", e);
		}
	}

	/** @return the decompressed version of the given (compressed with gzip) bytes */
	public static Object gunzipObject(byte[] in) {
		try {
			return new ObjectInputStream(new GZIPInputStream(new ByteArrayInputStream(in))).readObject();
		} catch (IOException | ClassNotFoundException e) {
			throw new RuntimeException("Failed to decompress data", e);
		}
	}

	/** ObjectInputStream which doesn't care too much about serialVersionUIDs. Horrible :)
	 */
	public static ObjectInputStream gullibleObjectInputStream(InputStream is) throws IOException {
		return new ObjectInputStream(is) {
			@Override
			protected ObjectStreamClass readClassDescriptor() throws IOException, ClassNotFoundException {
				ObjectStreamClass oc = super.readClassDescriptor();
				try {
					Class<?> c = Class.forName(oc.getName());
					// interfaces do not have fields
					if (!c.isInterface()) {
						Field f = oc.getClass().getDeclaredField("suid");
						f.setAccessible(true);
						f.set(oc, ObjectStreamClass.lookup(c).getSerialVersionUID());
					}
				} catch (Exception e) {
					System.err.println("Couldn't fake class descriptor for "+oc+": "+ e);
				}
				return oc;
			}
		};
	}

	/** Close, ignoring exceptions */
	public static void close(Closeable stream) {
		try {
			stream.close();
		} catch (IOException e) {			
		}
	}

	/**
	 * Creates a directory if it does not exist.
	 * 
	 * <p>
	 * It will recursively create all the absolute path specified in the input
	 * parameter.
	 * 
	 * @param directory
	 *            the {@link File} containing the directory structure that is
	 *            going to be created.
	 * @return a {@link File} pointing to the created directory
	 * @throws IOException if there was en error while creating the directory.
	 */
	public static File createDirIfNotExists(File directory) throws IOException {
		if (!directory.isDirectory()) {
			if (!directory.mkdirs()) {
				throw new IOException("Failed to create directory: " + directory.getAbsolutePath());
			}
		}
		return directory;
	}

	/**
	 * @return true if the {@link File} is null, does not exist, or we were able to delete it; false 
	 * if the file exists and could not be deleted.
	 */
	public static boolean deleteIfPresent(File f) {
		return f == null || !f.exists() || f.delete();
	}


	/** 
	 * Stores the given contents into a temporary file 
	 * @param fileContents the raw contents to store in the temporary file
	 * @param namePrefix the desired file name prefix (must be at least 3 characters long)
	 * @param extension the desired extension including the '.' character (use null for '.tmp')
	 * @return a {@link File} reference to the newly created temporary file
	 * @throws IOException if the temporary file creation fails 
	 */
	public static File createTempFile(byte[] fileContents, String namePrefix, String extension) throws IOException {
		Preconditions.checkNotNull(fileContents, "file contents missing");
		File tempFile = File.createTempFile(namePrefix, extension);
		try (FileOutputStream fos = new FileOutputStream(tempFile)) {
			fos.write(fileContents);
		}
		return tempFile;
	}
}
