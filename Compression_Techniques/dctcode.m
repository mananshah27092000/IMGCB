% creating DCT Matrix
T=zeros(8,8);
T(1,:)=ones(1,8)/sqrt(8);
for i=2:8
for j=1:8 
T(i,j)=sqrt(2/8)*cos((2*(j-1)+1)*(i-1)*pi/16);
end
end
Q=[16,11,10,16,24,40,51,61; 
12,12,14,19,26,58,60,55;
14,13,16,24,40,57,69,56;
14,17,22,29,51,87,80,62;
18,22,37,56,68,109,103,77;
24,35,55,64,81,104,113,92;
49,64,78,87,103,121,120,101;
72,92,95,98,112,100,103,99;
];
OrigIMG=imread('test.JPG');
IMG=rgb2ycbcr(OrigIMG);
%R=InpImg(:,:,1);
%G=InpImg(:,:,2);
%B=InpImg(:,:,3);
%Y  = 16 + 65.738*R/256 + 129.057*G/256 + 25.064*B/256 
%Cb = -37.945*R/256 - 74.494*G/256 + 112.439*B/256 
%Cr = 112.439*R/256 - 94.154*G/256 - 18.285*B/256
%IMG(:,:,1)=Y;
%IMG(:,:,2)=Cb;
%IMG(:,:,3)=Cr;
w=length(IMG(1,:,1)); 
h=length(IMG(:,1,1)); 
h=h-mod(h,8); 
w=w-mod(w,8);
hmax=h/8;
wmax=w/8;
IMG=IMG(1:h,1:w,:);
Yq= 1 % Luminance compression Ratio 
Cbq= 10 % Blue chrominance compression ratio
Crq= 10 % Red chrominance compression ratio
Qual=[Yq,Cbq,Crq];
% Initialising Compressed Image
finalIMG=zeros(h,w,3);
nonzero=zeros(2,3);
tic
for kk=1:3
I=IMG(:,:,kk);
M2 =double(I)-128;
C2 =zeros(h,w);
Q2 =Q*Qual(kk); % Adjusting Quantization Matrix for specific ratios 
for i=1:hmax
for j=1:wmax
M=M2(8*i-7:8*i,8*j-7:8*j);
D=T*M*T';
C=round(D./Q2); 
C2(8*i-7:8*i,8*j-7:8*j)=C;
end
end
R2=zeros(h,w);
for i=1:hmax
for j=1:wmax
R=C2(8*i-7:8*i,8*j-7:8*j).*Q2; 
R2(8*i-7:8*i,8*j-7:8*j)=R;
end
end
% IDCT
for i=1:hmax
for j=1:wmax
finalIMG(8*i-7:8*i,8*j-7:8*j,kk)=T'*R2(8*i-7:8*i,8*j-7:8*j)*T;
end
end
finalIMG(:,:,kk)=finalIMG(:,:,kk)+128;
nonzero(1,kk)=nnz(M2); % number of nonzero entries before compresssion
nonzero(2,kk)=nnz(C2); % number of nonzero entries after compresssion
end
finalIMG=real(uint8(finalIMG));
finalIMG=ycbcr2rgb(finalIMG);
toc
figure(2)
subplot(1,2,1)
imshow(OrigIMG)
title('Original')
subplot(1,2,2)
imshow(finalIMG)
title('Compressed')
nonzero

imwrite(finalIMG,'result2.jpg','jpg');