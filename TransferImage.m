%读取第r行以下r1行以上，l列以右l1列以左的区域    
%m=csvread('standardization.csv',r,l,[r,l,r1,l1]);

%读取第2行第3列之后的所有个数据
for i=1:50
    a=csvread('standardization.csv',i,2,[i,2,i,754]);
    d = zeros(28);
    for j=1:28
        for n=1:28
            if (n+28*(j-1)>752)
             break;
            else
                d(j,n)=a(n+28*(j-1));
            end
        end
    end
%imshow(d);
    filename=['pd_',int2str(i),'.jpg'];
    imwrite(d,filename,'jpg');
end
% 画出一行数据28x28，其余补0

%m=reshape(Data,[28,28]);
%imshow(Data);
%a1=reshape(m,28,28);
%imshow(a1);
%m;
