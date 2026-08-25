'''
in demo/ (pwd)

1. read the videos in demo_basketball_expert/
2. run wham for each video
3. inspect its output, and check the number of poses generated
4. if its more than 1, use a simple heuristic to select one
5; the heursitic: "movement"- select the pose that "moves" the most. This could be a term that is a sum of
the pelvis translation across frames.
    a) mid range jump shot- the pelvis moves vertically, shooting arm velocity/acceleration
    b) Mikan Layup- lower body should move?
    c) reverse layup- again lower body and shoulder?

these are some example hehuristics of what could be checked. or simply, translation of the pelvis,
which seems to be a unified heuristic could be better to start off with
'''

