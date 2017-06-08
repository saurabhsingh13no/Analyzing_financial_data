import logging
import sys

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

file_handler=logging.FileHandler('test.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}


# if len(sys.argv) > 1:
#     level_name = sys.argv[1]
#     level = LEVELS.get(level_name, logging.NOTSET)
#     logging.basicConfig(level=level)

logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical error message')

