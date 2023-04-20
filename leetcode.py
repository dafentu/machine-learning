import SimpleITK as sitk
import os
import numpy as np
import shutil
import csv


def findTargetSumWays(nums, target):
    cnt = 0
    n = len(nums)
    cnt += dfs(nums,target,0,0,n)
    return cnt

def dfs(nums,target,idx,sum,len):
    if sum == target and idx == len:
       return 1
    elif idx >= len:
        return 0
    else:
        return dfs(nums,target,idx+1,sum+nums[idx],len) + dfs(nums,target,idx+1,sum-nums[idx],len)

def canPartition(nums):
        n = len(nums)
        if n < 2:
            return False

        total = sum(nums)
        maxNum = max(nums)
        if total & 1:
            return False

        target = total // 2
        if maxNum > target:
            return False


        dp = [[False] * (target + 1) for _ in range(n)]
        for i in range(n):
            dp[i][0] = True
        dp[0][nums[0]] = True

        for i in range(n):
            for j in range(1,target+1):

                if j < nums[i]:

                    dp[i][j] = dp[i-1][j]
                else:

                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]

        return dp[n-1][target]




def wordBreak(s, wordDict):
    n = len(s)
    for str in wordDict:
        if df(0,s,wordDict,"",n) == True:
            return True

    return False

def df(idx,s,word,cur,n):
    if idx > n:
        return False
    if idx == n and cur == s:
        return True
    for str in word:
        if df(idx+len(str),s,word,cur+str,n) == True:
            return True
print(wordBreak("leetcode",["leet", "code"]))









#print(canPartition([1,5,11,5]))


















'''


def combinationSum4(nums, target):
   nums.sort()
   cnt = 0
   for idx,j in enumerate(nums):
       if j > target:
           break
       else:
           cnt += dfs(nums,target,j)
   return cnt
def dfs(nums,target,tmp):
    a = 0
    if tmp == target:
        return 1
    for idx, j in enumerate(nums):
        if tmp+j <= target:
            a += dfs(nums,target,tmp+j)
        else:
            break

    return a


import pandas as pd
import os
l1 = os.listdir("D:/in")
l2 = os.listdir("D:/out")
l3 = []
for i,j in zip(l1,l2):
    str1 = "/home/user/hym2/in/" + i
    str2 = "/home/user/hym2/out/" + j
    print(str1)
    l3.append((str1,str2))
name = ['Image','Mask']
test = pd.DataFrame(columns=name,data=l3)
test.to_csv(r'D:\2.csv')
'''








