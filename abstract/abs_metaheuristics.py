###############
#
#  
# Implements abstract class metaheuristic
#
###############

from abc import ABCMeta, abstractstaticmethod

# abstract class
class IMetaheuristic(metaclass=ABCMeta):
    @abstractstaticmethod
    def run():
        """ interface method """


