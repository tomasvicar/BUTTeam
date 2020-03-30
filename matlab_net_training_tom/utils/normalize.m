function data=normalize(data,means,stds)



for k=1:length(data)

    tmp=data{k};
    data{k}=(tmp-repmat(means,[1,size(tmp,2)]))./repmat(stds,[1,size(tmp,2)]);


end