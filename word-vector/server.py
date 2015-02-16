import web
import json
import gensim

web.config.debug = False
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

urls = (
	"/", "index"
)

class index:
	def POST(self):
		post_data = web.data()
		post_json = json.loads(post_data)
		positive = post_json['positive']
		negative = post_json['negative']
		# ret = {'pos': positive, 'neg': negative}
		ret = model.most_similar(positive=positive, negative=negative, topn=25)
		web.header('Content-Type', 'application/json')
		return json.dumps(ret)

if __name__ == "__main__":
	app = web.application(urls, globals())
	app.run()
