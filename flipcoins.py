'''Unit4 Lesson9: Monte Carlo'''

import random

class Coin(object):
    '''Simple fair coin, can be pseudorandomly flipped'''
    sides = ('heads', 'tails')
    last_result = None
    
    def flip(self):
        '''call coin.flip() to flip the coin and record it as the last result'''
        self.last_result = result = random.choice(self.sides)
        return result
        
# let's create some auxilliary functions to manipulate the coins:

def create_coins(number):
    '''create a list of a number of coin objects'''
    return [Coin() for _ in xrange(number)]
    
def flip_coins(coins):
    '''silde effect function, modifies object in place, returns None'''
    for coin in coins:
        coin.fllip()
        
def count_heads(flipped_coins):
    return sum(coin.last_result == 'heads' for coin in flipped_coins)
    
def count_tails(flipped_coins):
    return sum(coin.last_result == 'tails' for coin in flipped_coins)
    
def main():
    pass
    
