%��ȡ��r������r1�����ϣ�l������l1�����������    
%m=csvread('standardization.csv',r,l,[r,l,r1,l1]);

%��ȡ��2�е�3��֮������и�����
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
% ����һ������28x28�����ಹ0

%m=reshape(Data,[28,28]);
%imshow(Data);
%a1=reshape(m,28,28);
%imshow(a1);
%m;
