function morehot = more_hot_encode(lbls,pathologies)




morehot=repmat({zeros(1,length(pathologies))>0},[1,length(lbls)]);
for k = 1:length(lbls)
    lbl=lbls{k};
    
    
    
    res=zeros(1,length(pathologies),'single');
    
    pato=split(lbl,',');
    for kk=1:length(pato)
        tmp=strcmp(pathologies,pato{kk});
        res(tmp)=1;
    end
    morehot{k}=res;
end

end