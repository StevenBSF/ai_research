## 二维数组排序

```java
import java.util.Arrays;
import java.util.Comparator;
class cmp implements Comparator<int[]> {
    @Override
    public int compare(int[] o1,int[] o2){
        return o1[0] - o2[0];
    }
}
public class Main {
    public static void main(String[] args) {
        int[][] nums = {{1,7},{2,6},{3,9},{8,2},{2,5}};
        Arrays.sort(nums,new cmp());
        for(int i = 0;i < nums.length;i++){
            for(int j = 0;j < nums[i].length;j++) {
                System.out.print(nums[i][j] + " ");
            }
            System.out.println();
        }
    }
}

```

```c++
#include <stdio.h>
#include <stdlib.h>
int main()
{
   	
	int i,j,tmp,n,m;scanf("%d",&n);

	int **mainPtr=(int**)malloc(sizeof(int*)*n);
	for(i=0;i<n;i++)
	{
   	
		scanf("%d",&m);
		mainPtr[i]
```

```java
import java.util.Arrays;
import java.util.Scanner;
import java.util.Comparator;
class cmp implements Comparator<int[]> {
    @Override
    public int compare(int[] o1,int[] o2){
        return o1[0] - o2[0];
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

        int[][] nums = new int[200][1010];
        int n = input.nextInt();
        for(int i = 0;i < n;i++){
            nums[i][1] = input.nextInt();
            int m = nums[i][1];
            for(int j = 2;j <= m+1;j++){
                nums[i][j] = input.nextInt();
                nums[i][0] += nums[i][j];
            }
            nums[i][0] /= m;
        }
        Arrays.sort(nums,new cmp());
        for(int i = 0;i < n;i++){
            for(int j = 2;j <= nums[i][1]+1;j++){
                System.out.printf("%d",nums[i][j]);
                if(j < nums[i][1] +1){
                    System.out.printf(" ");
                }else{
                    System.out.printf("\n");
                }
            }
        }
    }

}
```

