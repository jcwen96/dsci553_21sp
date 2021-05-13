import sys, time, collections, random
import tweepy

API_KEY = "QNiFurxFwzAh1ie68giBEOOad"
API_SECRET_KEY = "wPnHj3edHCojKGVrutcnCsK8iQDilUSmEkbSP0yxn0Uf4ZkPFe"
ACCESS_TOKEN = "1258753240767709184-qYY7xbL8OyUQZwOr277lbIm99Dyq4C"
ACCESS_TOKEN_SECRET = "v00EBCnrTX55W32k7Od7mTQcjFaDLCe00bReBUy7vrszq"

TOPIC_LIST = ["COVID", "Trump", "Biden"]
SAMPLE_SIZE = 100

auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)


class TwitterStreamListener(tweepy.StreamListener):

    def __init__(self, output_file_path, api=None):
        super().__init__(api=api)
        self.output_file_path = output_file_path
        self.sequence_number = 0
        self.saved_tags = []
        self.tag_count = collections.defaultdict(int)

    def on_status(self, status):
        tag_dict_list = status.entities["hashtags"]

        if tag_dict_list:
            tag_list = list(map(lambda x: x["text"], tag_dict_list))
            print("=============DEBUG==============")
            print(tag_list)
            self._add_one_sequence(tag_list)
            print(self.sequence_number)
            print(len(self.saved_tags))

    def _add_one_sequence(self, tag_list):
        self.sequence_number += 1
        if self.sequence_number <= SAMPLE_SIZE:
            self.saved_tags.append(tag_list)
            for tag in tag_list:
                self.tag_count[tag] += 1
        else:
            # reservoir sampling
            # with probablity s/n, keep the nth element
            probablity = SAMPLE_SIZE / self.sequence_number
            if random.random() < probablity: # keep the nth element
                delete_tag_list = self.saved_tags.pop(random.randrange(len(self.saved_tags)))
                for tag in delete_tag_list:
                    self.tag_count[tag] -= 1
                self.saved_tags.append(tag_list)
                for tag in tag_list:
                    self.tag_count[tag] += 1
        self._output_res()
    
    def _output_res(self):
        results = sorted(list(self.tag_count.items()), key=lambda x: (-x[1], x[0]))
        with open(self.output_file_path, 'a', encoding="utf-8") as f:
            print("The number of tweets with tags from the beginning:", self.sequence_number, file = f)
            top_3_counts = set()
            for one in results:
                top_3_counts.add(one[1])
                if len(top_3_counts) <= 3:
                    print(one[0], ':', one[1], file = f)
                else:
                    break
            print(file = f)



if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    port_num = int(sys.argv[1])
    output_file_path = sys.argv[2]

    with open(output_file_path, 'w') as f:
        print()

    # Step 1: Creating a StreamListener
    my_stream_listener = TwitterStreamListener(output_file_path)
    # Step 2: Creating a Stream
    my_stream = tweepy.Stream(api.auth, my_stream_listener)
    # Step 3: Starting a Stream
    my_stream.filter(track=TOPIC_LIST, languages=["en"])
    # my_stream.sample(languages=["en"])
    

    print("Duration: {0:.2f}".format(time.time() - start_time))
