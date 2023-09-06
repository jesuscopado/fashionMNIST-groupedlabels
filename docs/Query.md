```SQL
SELECT 
    P.Title AS ProductTitle,
    I.Url AS ImageUrl,
    COALESCE(PD.TranslatedText, PD.OriginalText) AS DescriptionText
FROM 
    Product P
-- Left join with ProductImages table to ensure we get all products even if they don't have an image
LEFT JOIN ProductImages PI ON P.Id = PI.ProductId AND PI.ImageIndex = 0 
-- Left join with Image table to get the image URL
LEFT JOIN Image I ON PI.ImageId = I.Id
-- Left join with ProductDescription table to get the description for 'us' country code
LEFT JOIN ProductDescription PD ON P.Id = PD.ProductId AND PD.CountryCode = 'us'
```