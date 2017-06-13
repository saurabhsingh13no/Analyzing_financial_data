import logging
# Creating handlers for logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:'
                              '%(funcName)s:'
                              '%(lineno)d:'
                              '%(levelname)s:'
                              '%(name)s:'
                              '%(message)s')
file_handler = \
    logging.FileHandler('Financial_data_analysis.log')
# file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)