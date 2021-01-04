
<div style="text-align:center"><pre>
               @@@  ,@@@                     
              @@@@@&   ,@@                   
               @@@      ,&&&&&&&.            
                   &&&&&&&&&&&&&&&&&&&       
                &&&&&&&&&&&&&&&&&&&&&&&&&    
              &&&&&&&&&&&&&&&&&&&&&&&&&&&&&  
             &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& 
            &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            &&&&&&&&&&&&&&&&&/////////////// 
             /////////////////////////////// 
             ///%%%%%%%%%%%%%%%%%%%///////// 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%///// 
         (///........///////......../////%// 
        (///........../////........../////%/ 
        (///....@@..../////...@@...../////%/ 
         (////....../////////......//////%// 
             /////////////////////////////// 
             /////////////////////////////// 
             %%%%%%%%%%%%%%%%%(///////////// 
             ......(......|....../////////// 
   \\\     ,,.%....(......|......%////////// 
   ///´´´´´  ......(......|.....%/////////// 
             /////////////////////////////// 
                       ,/////////,           
</pre></div>

# Peasabot

Agent submission for the Coder One AI Sports Challenge 2019.
Creators: Artificial Incompetence.

## Strategy
This solution uses multiple map representations to calculate optimal positions for bomb placement and locate safe areas in the map.
Using this same information, a path is calculated to target positions on each turn.

## Submission
We will continue to work on the bot in the main branch, fix some bugs and try different things, that means that the main branch will be different from the code submitted in the competition.
To have you try our submitted code, we are keeping a submission branch, containing the state of the bot as it was submitted.

## Future goals.
Our initial idea for the bot was to preprocess the map to meaningful data that we could feed into a neural network to make decisions about which behaviour to follow at any point.
We plan to slowly try out different architectures and see if we can find a network smart enough to play the game successfully.
