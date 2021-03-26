import logging

logger = logging.getLogger('dna')
logger.setLevel(logging.DEBUG)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_ch = logging.FileHandler("dna.debug.log",mode='w')
_ch.setFormatter(_formatter)
logger.addHandler(_ch)

