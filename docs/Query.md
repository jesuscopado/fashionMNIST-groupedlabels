```SQL
SELECT 
    P.Title AS ProductTitle,
    I.Url AS ImageUrl,
    COALESCE(
        (SELECT PD.TranslatedText FROM ProductDescription PD WHERE PD.ProductId = P.Id AND PD.TranslatedText IS NOT NULL LIMIT 1),
        (SELECT PD.OriginalText FROM ProductDescription PD WHERE PD.ProductId = P.Id AND PD.CountryCode = 'us' LIMIT 1)
    ) AS DescriptionText
FROM 
    Product P
LEFT JOIN ProductImages PI ON P.Id = PI.ProductId AND PI.ImageIndex = 0 
LEFT JOIN Image I ON PI.ImageId = I.Id
```