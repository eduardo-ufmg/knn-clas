import classifier_pb2 as pb

def load_dataset(filename: str) -> pb.Dataset:
  """
  Load the dataset from a file.
  Args:
    filename: The name of the file to load the dataset from.
  Returns:
    The loaded dataset, or None if loading failed.
  """

  dataset = pb.Dataset()
  try:
    with open(filename, 'rb') as f:
      dataset.ParseFromString(f.read())
    return dataset
  except:
    return None

def load_support_samples(filename: str) -> pb.SupportSamples:
  """
  Load the support samples from a file.
  Args:
    filename: The name of the file to load the support samples from.
  Returns:
    The loaded support samples, or None if loading failed.
  """

  samples = pb.SupportSamples()
  try:
    with open(filename, 'rb') as f:
      samples.ParseFromString(f.read())
    return samples
  except:
    return None

def load_test_samples(filename: str) -> pb.TestSamples:
  """
  Load the test samples from a file.
  Args:
    filename: The name of the file to load the test samples from.
  Returns:
    The loaded test samples, or None if loading failed.
  """

  samples = pb.TestSamples()
  try:
    with open(filename, 'rb') as f:
      samples.ParseFromString(f.read())
    return samples
  except:
    return None

def load_predicted_samples(filename: str) -> pb.PredictedSamples:
  """
  Load the predicted samples from a file.
  Args:
    filename: The name of the file to load the predicted samples from.
  Returns:
    The loaded predicted samples, or None if loading failed.
  """

  samples = pb.PredictedSamples()
  try:
    with open(filename, 'rb') as f:
      samples.ParseFromString(f.read())
    return samples
  except:
    return None
  