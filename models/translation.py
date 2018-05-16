import math
import time
import sys

import pandas as pd
from nltk.tokenize import sent_tokenize
from google.cloud import translate
from google.api_core.exceptions import ServiceUnavailable, BadRequest
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local") \
    .appName("Data Load") \
    .getOrCreate()

log4jLogger = spark._jvm.org.apache.log4j
logger = log4jLogger.LogManager.getLogger(__name__)

logger.info('Translation started')

pravda = spark.read.json('s3a://vikua-news-data/pravda_news.json')
pravda_sample = pravda.where('lang = "uk"').sample(fraction=0.1, seed=1234).toPandas()

ob = spark.read.json('s3a://vikua-news-data/obozrevatel_economy.json')
ob_sample = ob.where('lang = "uk"').sample(fraction=0.2, seed=1234).toPandas()

pravda_sample['source'] = 'pravda_news'
ob_sample['source'] = 'obozrevatel_economy'
df = pd.concat([pravda_sample, ob_sample])

translate_client = translate.Client()

df['body'] = df['body'].apply(lambda x: sent_tokenize(x, language='russian'))


def translate_func(sents):
    try:
        return translate_client.translate(sents, source_language='uk', target_language='en')
    except ServiceUnavailable as e:
        logger.info('[exception] {}'.format(str(e)))
        return 'ServiceUnavailable'
    except BadRequest as e:
        logger.info('[exception] {}'.format(str(e)))
        return 'BadRequest'


batch_size = 1
batches = int(math.ceil(len(df) / float(batch_size)))

logger.info('Batches to translate - {}'.format(batches))
frames = []
for i in range(batches)[12000:]:
    d = df.iloc[i * batch_size:(i + 1) * batch_size, :]
    d['body_en'] = d['body'].apply(lambda x: translate_func(x))
    frames.append(d)
    logger.info('{} translated'.format(i))

    if i != 0 and i % 1000 == 0:
        pd.concat(frames).to_parquet('translated-{}.parquet'.format(i), compression='uncompressed')
        frames = []
        logger.info('Step {} results saved'.format(i))

    time.sleep(2)
logger.info('Done')

pd.concat(frames).to_parquet('translated.parquet', compression='uncompressed')

sys.exit(0)