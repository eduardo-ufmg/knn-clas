import classifier_pb2 as pb

def store_dataset(dataset: pb.Dataset, filename: str) -> bool:
  """
  Store the dataset in a file.
  Args:
    dataset: The dataset to store.
    filename: The name of the file to store the dataset in.
  Returns:
    True if the dataset was stored successfully, False otherwise.
  """
  try:
    with open(filename, 'wb') as f:
      f.write(dataset.SerializeToString())
    return True
  except:
    return False

def store_support_samples(samples: pb.SupportSamples, filename: str) -> bool:
  """
  Store the support samples in a file.
  Args:
    samples: The support samples to store.
    filename: The name of the file to store the support samples in.
  Returns:
    True if the support samples were stored successfully, False otherwise.
  """
  try:
    with open(filename, 'wb') as f:
      f.write(samples.SerializeToString())
    return True
  except:
    return False

def store_test_samples(samples: pb.TestSamples, filename: str) -> bool:
  """
  Store the test samples in a file.
  Args:
    samples: The test samples to store.
    filename: The name of the file to store the test samples in.
  Returns:
    True if the test samples were stored successfully, False otherwise.
  """
  try:
    with open(filename, 'wb') as f:
      f.write(samples.SerializeToString())
    return True
  except:
    return False

def store_predicted_samples(samples: pb.PredictedSamples, filename: str) -> bool:
  """
  Store the predicted samples in a file.
  Args:
    samples: The predicted samples to store.
    filename: The name of the file to store the predicted samples in.
  Returns:
    True if the predicted samples were stored successfully, False otherwise.
  """
  try:
    with open(filename, 'wb') as f:
      f.write(samples.SerializeToString())
    return True
  except:
    return False