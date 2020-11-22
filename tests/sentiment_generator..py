positive_group = {0,1,2,3,4}
negative_group = {5,6,7,8,9}
neutral_group = {9,10,11,12,13}

def get_sentiment(input):
  score = 0.0
  for i in range(len(input)-1,0,-1):
    v = input[i]
    if v in positive_group:
      score = score + 1.0
    elif v in negative_group:
      score = score * -1.0
  return 1.0 if score > 0.0 else 0.0
  
test_input = [1,3,10,11,6,5,4,3,1]
print(get_sentiment(test_input))
