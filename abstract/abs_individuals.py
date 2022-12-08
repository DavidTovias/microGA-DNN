###############
#
#  
# Implements abstract class individual
#
###############

from abc import ABCMeta, abstractstaticmethod

# abstract class
class IIndividual(metaclass=ABCMeta):
    @abstractstaticmethod
    def evaluate():
        """ interface method """


